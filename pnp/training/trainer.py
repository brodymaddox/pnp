"""
Training Pipeline for Pink Noise Prior Music Generation.

Implements the core training loop with:
- Cross-entropy loss for next-token prediction
- Spectral slope loss for fractal structure enforcement
- Mixed precision training support
- Gradient accumulation
- Comprehensive logging
"""

import os
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from pnp.models.transformer import MusicTransformer, MusicTransformerConfig
from pnp.losses.spectral import CombinedMusicLoss
from pnp.training.scheduler import WarmupCosineScheduler


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training basics
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Pink noise loss
    lambda_pink: float = 0.1
    target_slope: float = -1.0
    lambda_warmup_epochs: int = 5  # Gradually increase lambda

    # Learning rate schedule
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Mixed precision
    use_amp: bool = True

    # Logging and checkpointing
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "pnp_experiment"

    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "pink-noise-prior"

    # Reproducibility
    seed: int = 42


class Trainer:
    """
    Trainer for MusicTransformer with Pink Noise Prior.

    Handles the training loop, evaluation, checkpointing, and logging.

    Args:
        model: MusicTransformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: TrainingConfig
        tokenizer: Optional tokenizer for generation
    """

    def __init__(
        self,
        model: MusicTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        tokenizer=None,
    ):
        self.config = config or TrainingConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Combined loss function
        self.loss_fn = CombinedMusicLoss(
            lambda_pink=0.0,  # Start with 0, will warmup
            target_slope=self.config.target_slope,
            vocab_size=model.config.vocab_size,
        ).to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.config.min_lr_ratio,
        )

        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir) / self.config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        self.wandb_run = None
        if self.config.use_wandb:
            self._init_wandb()

    def _create_optimizer(self) -> torch.optim.AdamW:
        """Create AdamW optimizer with weight decay exclusions."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'ln' in name or 'LayerNorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
        )

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config={
                    'model': vars(self.model.config),
                    'training': vars(self.config),
                },
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            self.config.use_wandb = False

    def _get_current_lambda(self) -> float:
        """Get current lambda_pink with warmup."""
        if self.epoch < self.config.lambda_warmup_epochs:
            # Linear warmup
            progress = self.epoch / self.config.lambda_warmup_epochs
            return self.config.lambda_pink * progress
        return self.config.lambda_pink

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_ce_loss': [],
            'train_spectral_loss': [],
            'val_loss': [],
            'spectral_slope': [],
            'learning_rate': [],
        }

        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Update lambda for warmup
            current_lambda = self._get_current_lambda()
            self.loss_fn.set_lambda(current_lambda)

            # Train epoch
            train_metrics = self._train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_ce_loss'].append(train_metrics['ce_loss'])
            history['train_spectral_loss'].append(train_metrics['spectral_loss'])
            history['spectral_slope'].append(train_metrics['spectral_slope'])
            history['learning_rate'].append(self.scheduler.get_lr())

            # Validation
            if self.val_loader is not None:
                val_loss = self._validate()
                history['val_loss'].append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best.pt')

            # Epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(CE: {train_metrics['ce_loss']:.4f}, "
                  f"Spectral: {train_metrics['spectral_loss']:.4f})")
            print(f"  Spectral Slope: {train_metrics['spectral_slope']:.3f} "
                  f"(target: {self.config.target_slope})")
            print(f"  Lambda Pink: {current_lambda:.4f}")
            if self.val_loader:
                print(f"  Val Loss: {val_loss:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')

        # Final save
        self.save_checkpoint('final.pt')

        return history

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_spectral_loss = 0.0
        total_slope = 0.0
        n_batches = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, (input_ids, targets) in enumerate(progress):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                logits = self.model(input_ids)
                loss, components = self.loss_fn(
                    logits, targets, return_components=True
                )
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate metrics
            total_loss += components['total_loss'].item()
            total_ce_loss += components['ce_loss'].item()
            total_spectral_loss += components['spectral_loss'].item()
            total_slope += components['spectral_slope'].item()
            n_batches += 1

            # Update progress bar
            if batch_idx % self.config.log_every == 0:
                progress.set_postfix({
                    'loss': f"{components['total_loss'].item():.4f}",
                    'slope': f"{components['spectral_slope'].item():.2f}",
                    'lr': f"{self.scheduler.get_lr():.2e}",
                })

            # Wandb logging
            if self.config.use_wandb and batch_idx % self.config.log_every == 0:
                import wandb
                wandb.log({
                    'train/loss': components['total_loss'].item(),
                    'train/ce_loss': components['ce_loss'].item(),
                    'train/spectral_loss': components['spectral_loss'].item(),
                    'train/spectral_slope': components['spectral_slope'].item(),
                    'train/lambda_pink': self.loss_fn.lambda_pink,
                    'train/learning_rate': self.scheduler.get_lr(),
                    'train/step': self.global_step,
                })

        return {
            'loss': total_loss / n_batches,
            'ce_loss': total_ce_loss / n_batches,
            'spectral_loss': total_spectral_loss / n_batches,
            'spectral_slope': total_slope / n_batches,
        }

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            with autocast(enabled=self.config.use_amp):
                logits = self.model(input_ids)
                loss = self.loss_fn(logits, targets)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': vars(self.config),
            'model_config': vars(self.model.config),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }

        if self.config.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.config.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")

    @torch.no_grad()
    def generate_samples(
        self,
        n_samples: int = 5,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> List[torch.Tensor]:
        """Generate sample sequences for evaluation."""
        self.model.eval()

        samples = []
        for _ in range(n_samples):
            # Start with BOS token
            start_tokens = torch.tensor([[1]], device=self.device)

            generated = self.model.generate(
                start_tokens,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=2,  # EOS token
            )

            samples.append(generated.cpu())

        return samples


def train_baseline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: MusicTransformerConfig,
    training_config: TrainingConfig,
    **kwargs,
) -> Tuple[MusicTransformer, Dict]:
    """
    Train baseline model (CE loss only, lambda_pink=0).

    Args:
        train_loader: Training data
        val_loader: Validation data
        model_config: Model configuration
        training_config: Training configuration

    Returns:
        Trained model and training history
    """
    training_config.lambda_pink = 0.0
    training_config.experiment_name = f"{training_config.experiment_name}_baseline"

    model = MusicTransformer(model_config)
    trainer = Trainer(model, train_loader, val_loader, training_config, **kwargs)
    history = trainer.train()

    return model, history


def train_pink_noise_prior(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: MusicTransformerConfig,
    training_config: TrainingConfig,
    **kwargs,
) -> Tuple[MusicTransformer, Dict]:
    """
    Train model with Pink Noise Prior (CE + spectral loss).

    Args:
        train_loader: Training data
        val_loader: Validation data
        model_config: Model configuration
        training_config: Training configuration

    Returns:
        Trained model and training history
    """
    training_config.experiment_name = f"{training_config.experiment_name}_pink"

    model = MusicTransformer(model_config)
    trainer = Trainer(model, train_loader, val_loader, training_config, **kwargs)
    history = trainer.train()

    return model, history
