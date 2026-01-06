"""
Autocorrelation Analysis for Music Sequences.

Autocorrelation measures how much a sequence correlates with delayed
versions of itself. High-quality music exhibits long-range correlations
(structure preserved over long time scales).

Metrics:
- Autocorrelation decay rate
- Long-range correlation strength
- Characteristic time scale
"""

import numpy as np
import torch
from typing import Union, Tuple, List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class AutocorrelationResult:
    """Results from autocorrelation analysis."""
    decay_rate: float  # Exponential decay rate
    long_range_correlation: float  # Correlation at long lags
    characteristic_lag: int  # Lag where correlation drops to 1/e
    r_squared: float  # Quality of exponential fit
    acf_values: np.ndarray  # Full autocorrelation function


class AutocorrelationAnalyzer:
    """
    Analyze autocorrelation structure of music sequences.

    Good music typically shows:
    - Slow decay (structure preserved)
    - Significant long-range correlations
    - Large characteristic time scale

    Args:
        max_lag_ratio: Maximum lag as ratio of sequence length
        long_range_lag: Lag for measuring long-range correlation
    """

    def __init__(
        self,
        max_lag_ratio: float = 0.5,
        long_range_lag: int = 100,
    ):
        self.max_lag_ratio = max_lag_ratio
        self.long_range_lag = long_range_lag

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> AutocorrelationResult:
        """
        Compute autocorrelation metrics.

        Args:
            x: 1D time series

        Returns:
            AutocorrelationResult with all metrics
        """
        # Convert to numpy
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64).ravel()

        N = len(x)
        if N < 20:
            return AutocorrelationResult(
                decay_rate=np.nan,
                long_range_correlation=0.0,
                characteristic_lag=0,
                r_squared=0.0,
                acf_values=np.array([]),
            )

        # Compute autocorrelation
        max_lag = min(int(N * self.max_lag_ratio), N - 1)
        acf = self._compute_acf(x, max_lag)

        # Long-range correlation
        lr_lag = min(self.long_range_lag, max_lag - 1)
        long_range_corr = acf[lr_lag] if lr_lag < len(acf) else 0.0

        # Fit exponential decay to find characteristic scale
        decay_rate, char_lag, r_squared = self._fit_decay(acf)

        return AutocorrelationResult(
            decay_rate=decay_rate,
            long_range_correlation=long_range_corr,
            characteristic_lag=char_lag,
            r_squared=r_squared,
            acf_values=acf,
        )

    @staticmethod
    def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Compute normalized autocorrelation function.

        Uses FFT for efficiency.
        """
        N = len(x)

        # Mean-center the signal
        x = x - np.mean(x)

        # Variance for normalization
        var = np.var(x)
        if var < 1e-10:
            return np.zeros(max_lag)

        # FFT-based autocorrelation
        # Pad for linear (not circular) correlation
        n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
        fft_x = np.fft.fft(x, n=n_fft)
        acf_full = np.fft.ifft(fft_x * np.conj(fft_x)).real[:N]

        # Normalize
        acf = acf_full / (var * np.arange(N, 0, -1))

        return acf[:max_lag]

    def _fit_decay(
        self,
        acf: np.ndarray,
    ) -> Tuple[float, int, float]:
        """
        Fit exponential decay to ACF to find characteristic time scale.

        ACF(lag) ≈ exp(-λ * lag)
        log(ACF) ≈ -λ * lag

        Returns:
            decay_rate: λ (larger = faster decay = less structure)
            characteristic_lag: 1/λ (lag where ACF ≈ 1/e)
            r_squared: Quality of fit
        """
        # Skip lag 0 (always 1) and find valid positive values
        lags = np.arange(1, len(acf))
        values = acf[1:]

        # Only fit to positive values
        valid = values > 0.05  # Threshold for meaningful correlation
        if np.sum(valid) < 3:
            return 0.0, 0, 0.0

        log_acf = np.log(values[valid])
        valid_lags = lags[valid]

        # Linear regression in log space
        n = len(valid_lags)
        sum_x = np.sum(valid_lags)
        sum_y = np.sum(log_acf)
        sum_xy = np.sum(valid_lags * log_acf)
        sum_x2 = np.sum(valid_lags ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.0, 0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # Decay rate is negative of slope
        decay_rate = -slope

        # Characteristic lag
        if decay_rate > 0:
            char_lag = int(1.0 / decay_rate)
        else:
            char_lag = len(acf)  # Very slow decay

        # R-squared
        intercept = (sum_y - slope * sum_x) / n
        y_pred = slope * valid_lags + intercept
        ss_res = np.sum((log_acf - y_pred) ** 2)
        ss_tot = np.sum((log_acf - np.mean(log_acf)) ** 2)
        r_squared = max(0.0, 1 - ss_res / (ss_tot + 1e-10))

        return decay_rate, char_lag, r_squared

    def plot(
        self,
        x: Union[np.ndarray, torch.Tensor],
        ax: plt.Axes = None,
        title: str = "Autocorrelation Function",
    ) -> plt.Figure:
        """
        Plot the autocorrelation function.

        Args:
            x: Time series
            ax: Matplotlib axes
            title: Plot title

        Returns:
            Matplotlib figure
        """
        result = self(x)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        lags = np.arange(len(result.acf_values))

        # Plot ACF
        ax.plot(lags, result.acf_values, 'b-', linewidth=1.5, label='ACF')

        # Plot exponential fit
        if result.decay_rate > 0 and result.r_squared > 0.5:
            fit = np.exp(-result.decay_rate * lags)
            ax.plot(lags, fit, 'r--', linewidth=1.5,
                   label=f'Exp fit (τ={result.characteristic_lag})')

        # Mark long-range correlation point
        if result.long_range_correlation != 0:
            ax.axvline(self.long_range_lag, color='g', linestyle=':',
                      label=f'LR corr = {result.long_range_correlation:.3f}')

        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


def compute_autocorrelation(
    x: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """
    Convenience function to compute autocorrelation function.

    Args:
        x: Time series

    Returns:
        Autocorrelation values
    """
    analyzer = AutocorrelationAnalyzer()
    result = analyzer(x)
    return result.acf_values


def batch_analyze_autocorrelation(
    sequences: List,
) -> Dict:
    """
    Analyze multiple sequences and compute statistics.

    Args:
        sequences: List of sequences

    Returns:
        Dictionary with statistics
    """
    analyzer = AutocorrelationAnalyzer()
    decay_rates = []
    long_range_corrs = []
    char_lags = []

    for seq in sequences:
        result = analyzer(seq)
        if result.r_squared > 0.5:
            decay_rates.append(result.decay_rate)
            long_range_corrs.append(result.long_range_correlation)
            char_lags.append(result.characteristic_lag)

    if not decay_rates:
        return {
            'decay_rate_mean': np.nan,
            'long_range_correlation_mean': np.nan,
            'characteristic_lag_mean': np.nan,
            'n_valid': 0,
        }

    return {
        'decay_rate_mean': np.mean(decay_rates),
        'long_range_correlation_mean': np.mean(long_range_corrs),
        'characteristic_lag_mean': np.mean(char_lags),
        'n_valid': len(decay_rates),
    }


def has_long_range_structure(
    long_range_correlation: float,
    threshold: float = 0.1,
) -> bool:
    """Check if sequence has significant long-range structure."""
    return long_range_correlation > threshold
