"""
Zipf's Law Analysis for Music Sequences.

Zipf's law states that in many natural datasets, the frequency of an
element is inversely proportional to its rank. High-quality music
typically follows Zipf's law with exponent around 1.0.

Zipf coefficient = -slope of log(rank) vs log(frequency)

Expected values:
- Random music: ~0 (flat distribution)
- High-quality music: 0.8-1.2 (Zipfian distribution)
"""

import numpy as np
import torch
from typing import Union, Tuple, List, Dict
from collections import Counter
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ZipfResult:
    """Results from Zipf analysis."""
    coefficient: float  # Zipf exponent (slope magnitude)
    r_squared: float  # Quality of fit
    unique_elements: int
    total_elements: int


class ZipfAnalyzer:
    """
    Analyze Zipf's law characteristics of music sequences.

    Zipf's law: frequency ∝ 1/rank^α

    In log-log space: log(freq) = -α * log(rank) + c

    Args:
        min_rank: Minimum rank to include (skip most common)
        max_rank_ratio: Maximum rank as ratio of unique elements
    """

    def __init__(
        self,
        min_rank: int = 1,
        max_rank_ratio: float = 0.9,
    ):
        self.min_rank = min_rank
        self.max_rank_ratio = max_rank_ratio

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor, List],
    ) -> ZipfResult:
        """
        Compute Zipf coefficient.

        Args:
            x: Sequence of elements (e.g., pitch values)

        Returns:
            ZipfResult with coefficient and fit quality
        """
        # Convert to list
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().tolist()
        elif isinstance(x, np.ndarray):
            x = x.tolist()

        # Count frequencies
        counts = Counter(x)
        total = len(x)
        n_unique = len(counts)

        if n_unique < 3:
            return ZipfResult(
                coefficient=0.0,
                r_squared=0.0,
                unique_elements=n_unique,
                total_elements=total,
            )

        # Sort by frequency (descending)
        sorted_counts = sorted(counts.values(), reverse=True)

        # Create rank-frequency pairs
        ranks = np.arange(1, len(sorted_counts) + 1)
        frequencies = np.array(sorted_counts)

        # Apply range limits
        max_rank = max(int(n_unique * self.max_rank_ratio), self.min_rank + 2)
        valid = (ranks >= self.min_rank) & (ranks <= max_rank) & (frequencies > 0)

        if np.sum(valid) < 3:
            return ZipfResult(
                coefficient=0.0,
                r_squared=0.0,
                unique_elements=n_unique,
                total_elements=total,
            )

        log_ranks = np.log(ranks[valid])
        log_freqs = np.log(frequencies[valid])

        # Linear regression
        slope, _, r_squared = self._linear_regression(log_ranks, log_freqs)

        # Zipf coefficient is magnitude of negative slope
        coefficient = -slope

        return ZipfResult(
            coefficient=coefficient,
            r_squared=r_squared,
            unique_elements=n_unique,
            total_elements=total,
        )

    @staticmethod
    def _linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Compute linear regression and R²."""
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.0, 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return slope, intercept, max(0.0, r_squared)

    def plot(
        self,
        x: Union[np.ndarray, torch.Tensor, List],
        ax: plt.Axes = None,
        title: str = "Zipf Distribution",
    ) -> plt.Figure:
        """
        Plot the rank-frequency distribution.

        Args:
            x: Sequence of elements
            ax: Matplotlib axes (created if None)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().tolist()
        elif isinstance(x, np.ndarray):
            x = x.tolist()

        counts = Counter(x)
        sorted_counts = sorted(counts.values(), reverse=True)
        ranks = np.arange(1, len(sorted_counts) + 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Plot data
        ax.loglog(ranks, sorted_counts, 'o', alpha=0.7, label='Data')

        # Fit line
        result = self(x)
        if result.r_squared > 0.5:
            log_ranks = np.log(ranks)
            log_fit = -result.coefficient * log_ranks + np.log(sorted_counts[0])
            ax.loglog(ranks, np.exp(log_fit), '--',
                     label=f'Fit (α={result.coefficient:.2f}, R²={result.r_squared:.2f})')

        ax.set_xlabel('Rank')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


def compute_zipf_coefficient(
    x: Union[np.ndarray, torch.Tensor, List],
) -> float:
    """
    Convenience function to compute Zipf coefficient.

    Args:
        x: Sequence of elements

    Returns:
        Zipf coefficient (exponent)
    """
    analyzer = ZipfAnalyzer()
    result = analyzer(x)
    return result.coefficient


def batch_analyze_zipf(
    sequences: List,
) -> Dict:
    """
    Analyze multiple sequences and compute Zipf statistics.

    Args:
        sequences: List of sequences to analyze

    Returns:
        Dictionary with mean, std for Zipf coefficient
    """
    analyzer = ZipfAnalyzer()
    coefficients = []

    for seq in sequences:
        result = analyzer(seq)
        if result.r_squared > 0.5:
            coefficients.append(result.coefficient)

    if not coefficients:
        return {
            'zipf_coefficient_mean': np.nan,
            'zipf_coefficient_std': np.nan,
            'n_valid': 0,
        }

    return {
        'zipf_coefficient_mean': np.mean(coefficients),
        'zipf_coefficient_std': np.std(coefficients),
        'n_valid': len(coefficients),
    }


def is_zipfian(
    coefficient: float,
    min_coef: float = 0.7,
    max_coef: float = 1.5,
) -> bool:
    """Check if distribution follows Zipf's law."""
    return min_coef <= coefficient <= max_coef
