"""
Configuration utilities for experiment management.

Supports YAML configuration files with nested structures
and command-line overrides.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}


def save_config(config: Dict[str, Any], path: str | Path):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two config dictionaries.

    Override values take precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config: Dict, prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested config to dot-separated keys.

    Example: {'model': {'layers': 6}} -> {'model.layers': 6}
    """
    result = {}
    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_config(value, new_key))
        else:
            result[new_key] = value
    return result


def unflatten_config(flat_config: Dict[str, Any]) -> Dict:
    """
    Unflatten dot-separated keys to nested dict.

    Example: {'model.layers': 6} -> {'model': {'layers': 6}}
    """
    result = {}
    for key, value in flat_config.items():
        parts = key.split('.')
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Experiment info
    name: str = "pnp_experiment"
    seed: int = 42
    output_dir: str = "outputs"

    # Data
    data_dir: str = "data/maestro"
    seq_length: int = 512
    batch_size: int = 32
    num_workers: int = 4

    # Model
    model_size: str = "small"  # tiny, small, medium, large
    vocab_size: int = 92
    max_seq_length: int = 512

    # Training
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 1000

    # Pink Noise Prior
    use_pink_loss: bool = True
    lambda_pink: float = 0.1
    target_slope: float = -1.0
    lambda_warmup_epochs: int = 5

    # Logging
    use_wandb: bool = False
    wandb_project: str = "pink-noise-prior"
    log_every: int = 100
    eval_every: int = 1000

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config: Dict) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ExperimentConfig':
        """Load from YAML file."""
        config = load_config(path)
        return cls.from_dict(config)

    def save(self, path: str | Path):
        """Save to YAML file."""
        save_config(self.to_dict(), path)


def create_default_config(output_path: str = "configs/default.yaml"):
    """Create and save default configuration file."""
    config = ExperimentConfig()
    config.save(output_path)
    return config
