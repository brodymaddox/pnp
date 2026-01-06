"""
Learning Rate Schedulers for Training.

Implements warmup + cosine annealing schedule commonly used
for training transformers.
"""

import math
from typing import Optional


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.

    During warmup: lr increases linearly from 0 to base_lr
    After warmup: lr follows cosine decay to min_lr

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of base LR
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # Store base learning rates for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        lr = self._get_lr(self.current_step)

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr * base_lr / self.base_lrs[0]

    def _get_lr(self, step: int) -> float:
        """Compute learning rate for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * step / max(1, self.warmup_steps)
        elif step >= self.total_steps:
            # After training, use minimum LR
            return self.base_lrs[0] * self.min_lr_ratio
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
            return self.base_lrs[0] * decayed

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']


class LinearWarmupScheduler:
    """
    Simple linear warmup scheduler.

    Increases LR linearly from 0 to base_lr over warmup_steps,
    then keeps constant.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr_scale = min(1.0, self.current_step / max(1, self.warmup_steps))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> dict:
        return {'current_step': self.current_step, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state_dict: dict):
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']


class ConstantScheduler:
    """Constant learning rate (no scheduling)."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict):
        pass
