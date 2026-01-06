"""
Fractal Dimension and Spectral Analysis Metrics.

This module implements metrics for analyzing the fractal/self-similar
structure of music sequences:

1. Higuchi Fractal Dimension (HFD) - measures complexity/irregularity
2. Spectral Slope - measures the 1/f characteristics

High-quality music typically has:
- HFD around 1.3-1.5 (complex but not random)
- Spectral slope around -1.0 (pink noise characteristics)
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FractalAnalysisResult:
    """Results from fractal analysis."""
    fractal_dimension: float
    spectral_slope: float
    r_squared: float  # Quality of linear fit
    is_valid: bool


class HiguchiFractalDimension:
    """
    Compute Higuchi Fractal Dimension (HFD) of a time series.

    The Higuchi method estimates the fractal dimension directly in the
    time domain, making it suitable for relatively short sequences.

    Reference: Higuchi, T. (1988). Approach to an irregular time series
    on the basis of the fractal theory.

    Expected values for music:
    - Random noise: ~2.0
    - High-quality music: 1.3-1.5
    - Very predictable: ~1.0

    Args:
        k_max: Maximum scale parameter (default: auto-computed)
    """

    def __init__(self, k_max: Optional[int] = None):
        self.k_max = k_max

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[float, float]:
        """
        Compute Higuchi Fractal Dimension.

        Args:
            x: 1D time series

        Returns:
            fd: Fractal dimension estimate
            r_squared: R² of the linear fit (higher = more reliable)
        """
        # Convert to numpy
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64).ravel()

        N = len(x)
        if N < 10:
            return np.nan, 0.0

        # Determine k_max
        k_max = self.k_max if self.k_max is not None else max(4, N // 4)
        k_max = min(k_max, N // 4)

        # Compute length for each scale k
        lengths = []
        scales = list(range(1, k_max + 1))

        for k in scales:
            L_k = 0
            for m in range(1, k + 1):
                # Construct sub-series starting at m with step k
                indices = np.arange(m - 1, N, k)
                sub_series = x[indices]

                if len(sub_series) < 2:
                    continue

                # Compute normalized length
                L_m = np.sum(np.abs(np.diff(sub_series)))
                n_segments = len(sub_series) - 1
                if n_segments > 0:
                    L_m = L_m * (N - 1) / (k * k * n_segments)
                    L_k += L_m

            if k > 0:
                L_k /= k
            lengths.append(L_k)

        lengths = np.array(lengths)

        # Filter out zeros for log
        valid = lengths > 0
        if np.sum(valid) < 2:
            return np.nan, 0.0

        log_scales = np.log(np.array(scales)[valid])
        log_lengths = np.log(lengths[valid])

        # Linear regression in log-log space
        # FD = -slope
        slope, intercept, r_squared = self._linear_regression(log_scales, log_lengths)

        fd = -slope
        return fd, r_squared

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

        return slope, intercept, r_squared


def compute_fractal_dimension(
    x: Union[np.ndarray, torch.Tensor],
    k_max: Optional[int] = None,
) -> float:
    """
    Convenience function to compute Higuchi Fractal Dimension.

    Args:
        x: 1D time series
        k_max: Maximum scale parameter

    Returns:
        Fractal dimension estimate
    """
    hfd = HiguchiFractalDimension(k_max=k_max)
    fd, _ = hfd(x)
    return fd


def compute_spectral_slope(
    x: Union[np.ndarray, torch.Tensor],
    min_freq_idx: int = 1,
    max_freq_ratio: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute the spectral slope of a sequence.

    The spectral slope indicates the 1/f^β characteristics:
    - β ≈ 0: White noise (no correlation)
    - β ≈ 1: Pink noise (ideal for music)
    - β ≈ 2: Brownian noise (too smooth)

    Args:
        x: 1D time series
        min_freq_idx: Minimum frequency index (skip DC)
        max_freq_ratio: Maximum frequency as ratio of Nyquist

    Returns:
        slope: Spectral slope (negative for 1/f signals)
        r_squared: Quality of linear fit
    """
    # Convert to numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).ravel()

    N = len(x)
    if N < 10:
        return 0.0, 0.0

    # Compute FFT
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2

    # Frequency range
    n_freqs = len(power)
    max_freq_idx = max(int(n_freqs * max_freq_ratio), min_freq_idx + 2)

    power = power[min_freq_idx:max_freq_idx]
    freqs = np.arange(min_freq_idx, min_freq_idx + len(power))

    # Filter zeros
    valid = power > 0
    if np.sum(valid) < 3:
        return 0.0, 0.0

    log_freqs = np.log(freqs[valid])
    log_power = np.log(power[valid])

    # Linear regression
    slope, _, r_squared = HiguchiFractalDimension._linear_regression(log_freqs, log_power)

    return slope, r_squared


def analyze_sequence(
    x: Union[np.ndarray, torch.Tensor],
) -> FractalAnalysisResult:
    """
    Perform complete fractal analysis of a sequence.

    Args:
        x: 1D time series (pitch sequence)

    Returns:
        FractalAnalysisResult with all metrics
    """
    fd, fd_r2 = HiguchiFractalDimension()(x)
    slope, slope_r2 = compute_spectral_slope(x)

    is_valid = not np.isnan(fd) and fd_r2 > 0.5 and slope_r2 > 0.5

    return FractalAnalysisResult(
        fractal_dimension=fd,
        spectral_slope=slope,
        r_squared=min(fd_r2, slope_r2),
        is_valid=is_valid,
    )


def batch_analyze_sequences(
    sequences: list,
) -> dict:
    """
    Analyze multiple sequences and compute statistics.

    Args:
        sequences: List of pitch sequences

    Returns:
        Dictionary with mean, std for each metric
    """
    fds = []
    slopes = []

    for seq in sequences:
        result = analyze_sequence(seq)
        if result.is_valid:
            fds.append(result.fractal_dimension)
            slopes.append(result.spectral_slope)

    if not fds:
        return {
            'fractal_dimension_mean': np.nan,
            'fractal_dimension_std': np.nan,
            'spectral_slope_mean': np.nan,
            'spectral_slope_std': np.nan,
            'n_valid': 0,
        }

    return {
        'fractal_dimension_mean': np.mean(fds),
        'fractal_dimension_std': np.std(fds),
        'spectral_slope_mean': np.mean(slopes),
        'spectral_slope_std': np.std(slopes),
        'n_valid': len(fds),
    }


def is_pink_noise(
    slope: float,
    tolerance: float = 0.3,
) -> bool:
    """Check if spectral slope indicates pink noise characteristics."""
    return abs(slope - (-1.0)) < tolerance


def is_musical_fractal_dimension(
    fd: float,
    min_fd: float = 1.2,
    max_fd: float = 1.6,
) -> bool:
    """Check if fractal dimension is in typical musical range."""
    return min_fd <= fd <= max_fd
