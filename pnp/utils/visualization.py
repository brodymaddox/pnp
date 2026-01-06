"""
Visualization utilities for training analysis and results.

Provides functions for plotting:
- Training curves (loss, metrics)
- Spectral analysis (PSD, slopes)
- Comparative analysis (baseline vs. pink noise prior)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """
    Plot training curves including loss and spectral metrics.

    Args:
        history: Dictionary with lists of metric values
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train', color='blue')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CE vs Spectral loss
    ax = axes[0, 1]
    if 'train_ce_loss' in history:
        ax.plot(history['train_ce_loss'], label='CE Loss', color='green')
    if 'train_spectral_loss' in history:
        ax.plot(history['train_spectral_loss'], label='Spectral Loss', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spectral slope
    ax = axes[1, 0]
    if 'spectral_slope' in history:
        ax.plot(history['spectral_slope'], label='Measured', color='purple')
        ax.axhline(-1.0, color='black', linestyle='--', label='Target (-1.0)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spectral Slope')
    ax.set_title('Spectral Slope (Pink Noise = -1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    if 'learning_rate' in history:
        ax.plot(history['learning_rate'], color='teal')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_spectral_analysis(
    sequence: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Spectral Analysis",
) -> plt.Figure:
    """
    Plot spectral analysis of a pitch sequence.

    Shows:
    - The sequence itself
    - Power spectral density
    - Log-log PSD with slope fit

    Args:
        sequence: 1D pitch sequence
        save_path: Optional save path
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from pnp.metrics.fractal import compute_spectral_slope

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original sequence
    ax = axes[0, 0]
    ax.plot(sequence, linewidth=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Pitch')
    ax.set_title('Pitch Sequence')
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[0, 1]
    ax.hist(sequence, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pitch')
    ax.set_ylabel('Frequency')
    ax.set_title('Pitch Distribution')
    ax.grid(True, alpha=0.3)

    # Power spectral density
    ax = axes[1, 0]
    fft = np.fft.rfft(sequence)
    power = np.abs(fft) ** 2
    freqs = np.arange(len(power))
    ax.plot(freqs[1:], power[1:])
    ax.set_xlabel('Frequency Index')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectral Density')
    ax.grid(True, alpha=0.3)

    # Log-log PSD with slope
    ax = axes[1, 1]
    valid = power[1:] > 0
    log_freqs = np.log(freqs[1:][valid])
    log_power = np.log(power[1:][valid])
    ax.scatter(log_freqs, log_power, alpha=0.5, s=10)

    # Fit line
    slope, r2 = compute_spectral_slope(sequence)
    x_fit = np.linspace(log_freqs.min(), log_freqs.max(), 100)
    y_fit = slope * x_fit + (log_power.mean() - slope * log_freqs.mean())
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Slope: {slope:.2f}')

    ax.set_xlabel('log(Frequency)')
    ax.set_ylabel('log(Power)')
    ax.set_title(f'Log-Log PSD (Slope = {slope:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison(
    baseline_history: Dict[str, List[float]],
    pink_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison between baseline and pink noise prior models.

    Args:
        baseline_history: Training history for baseline
        pink_history: Training history for pink noise prior
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss comparison
    ax = axes[0]
    if 'train_loss' in baseline_history:
        ax.plot(baseline_history['train_loss'], label='Baseline', color='blue')
    if 'train_loss' in pink_history:
        ax.plot(pink_history['train_loss'], label='Pink Prior', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spectral slope comparison
    ax = axes[1]
    if 'spectral_slope' in baseline_history:
        ax.plot(baseline_history['spectral_slope'], label='Baseline', color='blue')
    if 'spectral_slope' in pink_history:
        ax.plot(pink_history['spectral_slope'], label='Pink Prior', color='red')
    ax.axhline(-1.0, color='black', linestyle='--', label='Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spectral Slope')
    ax.set_title('Spectral Slope Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[2]
    if 'val_loss' in baseline_history:
        ax.plot(baseline_history['val_loss'], label='Baseline', color='blue')
    if 'val_loss' in pink_history:
        ax.plot(pink_history['val_loss'], label='Pink Prior', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_evaluation_metrics(
    baseline_metrics: Dict[str, float],
    pink_metrics: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot evaluation metrics comparison (bar chart).

    Args:
        baseline_metrics: Metrics for baseline model
        pink_metrics: Metrics for pink noise prior model
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    # Find common metrics
    common_keys = set(baseline_metrics.keys()) & set(pink_metrics.keys())
    # Filter to numeric values
    metrics = [k for k in common_keys if isinstance(baseline_metrics[k], (int, float))]

    if not metrics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No common metrics to compare',
                ha='center', va='center')
        return fig

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    baseline_vals = [baseline_metrics[m] for m in metrics]
    pink_vals = [pink_metrics[m] for m in metrics]

    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
    ax.bar(x + width/2, pink_vals, width, label='Pink Prior', color='red', alpha=0.7)

    ax.set_ylabel('Value')
    ax.set_title('Evaluation Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_experiment_report(
    output_dir: str,
    baseline_history: Dict,
    pink_history: Dict,
    baseline_metrics: Dict,
    pink_metrics: Dict,
):
    """
    Create a complete experiment report with all visualizations.

    Saves multiple plots to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    plot_training_curves(baseline_history, output_dir / 'baseline_training.png', 'Baseline Training')
    plot_training_curves(pink_history, output_dir / 'pink_training.png', 'Pink Prior Training')

    # Comparison
    plot_comparison(baseline_history, pink_history, output_dir / 'comparison.png')

    # Evaluation metrics
    plot_evaluation_metrics(baseline_metrics, pink_metrics, output_dir / 'evaluation.png')

    print(f"Report saved to {output_dir}")
