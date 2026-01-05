#!/usr/bin/env python
"""
Training script for Pink Noise Prior music generation.

Usage:
    python -m pnp.scripts.train --config configs/default.yaml
    python -m pnp.scripts.train --lambda-pink 0.1 --epochs 50

Experiments:
    Baseline (CE only):     --lambda-pink 0.0
    Pink Noise Prior:       --lambda-pink 0.1 (default)
    Brownian Prior:         --lambda-pink 0.1 --target-slope -2.0
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from pnp.data.tokenizer import MidiTokenizer
from pnp.data.maestro import MAESTRODataset, create_maestro_dataloaders, download_maestro
from pnp.models.transformer import MusicTransformer, MusicTransformerConfig, create_model
from pnp.training.trainer import Trainer, TrainingConfig
from pnp.utils.config import load_config, ExperimentConfig
from pnp.utils.visualization import plot_training_curves


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train music generation model with Pink Noise Prior"
    )

    # Config file
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )

    # Data
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Path to MAESTRO dataset'
    )
    parser.add_argument(
        '--download-data', action='store_true',
        help='Download MAESTRO dataset if not present'
    )
    parser.add_argument(
        '--max-files', type=int, default=None,
        help='Limit number of MIDI files (for debugging)'
    )

    # Model
    parser.add_argument(
        '--model-size', type=str, default='small',
        choices=['tiny', 'small', 'medium', 'large'],
        help='Model size preset'
    )
    parser.add_argument(
        '--seq-length', type=int, default=512,
        help='Sequence length for training'
    )

    # Training
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--gradient-clip', type=float, default=1.0,
        help='Gradient clipping value'
    )

    # Pink Noise Prior
    parser.add_argument(
        '--lambda-pink', type=float, default=0.1,
        help='Weight for spectral slope loss (0 = baseline)'
    )
    parser.add_argument(
        '--target-slope', type=float, default=-1.0,
        help='Target spectral slope (-1 = pink, -2 = brownian)'
    )
    parser.add_argument(
        '--lambda-warmup', type=int, default=5,
        help='Epochs to warmup lambda_pink'
    )

    # Output
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Experiment name (auto-generated if not specified)'
    )

    # Logging
    parser.add_argument(
        '--wandb', action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project', type=str, default='pink-noise-prior',
        help='W&B project name'
    )

    # Other
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--num-workers', type=int, default=0,
        help='DataLoader workers'
    )
    parser.add_argument(
        '--no-amp', action='store_true',
        help='Disable mixed precision training'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()

    # Override with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.model_size:
        config.model_size = args.model_size
    if args.seq_length:
        config.seq_length = args.seq_length
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.lambda_pink is not None:
        config.lambda_pink = args.lambda_pink
    if args.target_slope:
        config.target_slope = args.target_slope
    if args.lambda_warmup:
        config.lambda_warmup_epochs = args.lambda_warmup
    if args.wandb:
        config.use_wandb = True
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.seed:
        config.seed = args.seed

    # Generate experiment name
    if args.experiment_name:
        config.name = args.experiment_name
    else:
        mode = "baseline" if config.lambda_pink == 0 else f"pink_{config.lambda_pink}"
        config.name = f"{config.model_size}_{mode}_s{config.seed}"

    # Set random seed
    set_seed(config.seed)

    print("=" * 60)
    print("Pink Noise Prior - Music Generation Training")
    print("=" * 60)
    print(f"Experiment: {config.name}")
    print(f"Lambda Pink: {config.lambda_pink}")
    print(f"Target Slope: {config.target_slope}")
    print(f"Model Size: {config.model_size}")
    print(f"Epochs: {config.epochs}")
    print("=" * 60)

    # Setup data
    data_dir = Path(config.data_dir)
    if args.download_data or not data_dir.exists():
        print("Downloading MAESTRO dataset...")
        data_dir = download_maestro()

    # Create tokenizer
    tokenizer = MidiTokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_maestro_dataloaders(
        data_dir,
        tokenizer=tokenizer,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        max_files=args.max_files,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        size=config.model_size,
        max_seq_length=config.seq_length,
    )
    print(f"Model parameters: {model.get_num_params():,}")

    # Training config
    training_config = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gradient_clip=args.gradient_clip,
        lambda_pink=config.lambda_pink,
        target_slope=config.target_slope,
        lambda_warmup_epochs=config.lambda_warmup_epochs,
        use_amp=not args.no_amp,
        checkpoint_dir=args.output_dir,
        experiment_name=config.name,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        seed=config.seed,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        tokenizer=tokenizer,
    )

    # Train
    history = trainer.train()

    # Save training curves
    output_path = Path(args.output_dir) / config.name
    plot_training_curves(history, output_path / 'training_curves.png')

    # Save config
    config.save(output_path / 'config.yaml')

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
