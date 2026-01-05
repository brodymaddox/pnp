"""
Differentiable Spectral Slope Loss for Pink Noise Prior

This module implements the core innovation: a differentiable loss function that
enforces 1/f (pink noise) spectral characteristics in generated music sequences.

The key insight is treating the soft (differentiable) pitch predictions as a
continuous signal and applying spectral analysis during backpropagation.
"""

import torch
import torch.nn as nn
import torch.fft
from typing import Optional, Tuple
import math


class SpectralSlopeLoss(nn.Module):
    """
    Differentiable Spectral Slope Loss.

    Computes the power spectral density of a pitch sequence and penalizes
    deviation from a target spectral slope (default: -1.0 for pink noise).

    The loss is computed as:
        L_pink = || slope_generated - target_slope ||^2

    where the slope is estimated via linear regression in log-log space.

    Args:
        target_slope: Target spectral slope (default: -1.0 for pink noise)
                     -2.0 = Brownian noise (smoother, more predictable)
                     -1.0 = Pink noise (natural, musical structure)
                      0.0 = White noise (random, no structure)
        min_freq_idx: Minimum frequency index to consider (avoids DC component)
        max_freq_ratio: Maximum frequency as ratio of Nyquist (avoids aliasing artifacts)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        target_slope: float = -1.0,
        min_freq_idx: int = 1,
        max_freq_ratio: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.target_slope = target_slope
        self.min_freq_idx = min_freq_idx
        self.max_freq_ratio = max_freq_ratio
        self.eps = eps

    def forward(
        self,
        pitch_sequence: torch.Tensor,
        return_slope: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the spectral slope loss.

        Args:
            pitch_sequence: Tensor of shape (batch_size, seq_len) or (seq_len,)
                           containing pitch values (can be soft/continuous)
            return_slope: If True, also return the computed slope

        Returns:
            loss: Scalar loss tensor
            slope: (optional) Computed slope if return_slope=True
        """
        # Ensure batch dimension
        if pitch_sequence.dim() == 1:
            pitch_sequence = pitch_sequence.unsqueeze(0)

        batch_size, seq_len = pitch_sequence.shape

        # Compute FFT
        # Using rfft for real-valued input (more efficient)
        fft_result = torch.fft.rfft(pitch_sequence, dim=-1)

        # Power Spectral Density (magnitude squared)
        power_spectrum = torch.abs(fft_result) ** 2

        # Determine frequency range
        n_freqs = power_spectrum.shape[-1]
        max_freq_idx = max(int(n_freqs * self.max_freq_ratio), self.min_freq_idx + 2)

        # Extract relevant frequency range (avoid DC and high frequencies)
        power_spectrum = power_spectrum[:, self.min_freq_idx:max_freq_idx]
        n_points = power_spectrum.shape[-1]

        if n_points < 3:
            # Not enough points for meaningful slope estimation
            if return_slope:
                return torch.tensor(0.0, device=pitch_sequence.device), torch.tensor(0.0, device=pitch_sequence.device)
            return torch.tensor(0.0, device=pitch_sequence.device)

        # Create frequency axis (starting from min_freq_idx)
        freqs = torch.arange(
            self.min_freq_idx,
            self.min_freq_idx + n_points,
            dtype=pitch_sequence.dtype,
            device=pitch_sequence.device,
        )

        # Log-log transformation
        log_freqs = torch.log(freqs)
        log_power = torch.log(power_spectrum + self.eps)

        # Compute slope via closed-form linear regression
        # y = mx + b, solve for m
        # m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        slope = self._compute_slope(log_freqs, log_power)

        # Mean slope across batch
        mean_slope = slope.mean()

        # Loss: squared deviation from target slope
        loss = (mean_slope - self.target_slope) ** 2

        if return_slope:
            return loss, mean_slope
        return loss

    def _compute_slope(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute slope via differentiable linear regression.

        Args:
            x: Independent variable (log frequencies), shape (n_points,)
            y: Dependent variable (log power), shape (batch, n_points)

        Returns:
            slopes: Tensor of slopes for each batch element
        """
        n = x.shape[0]

        # Expand x for batch computation
        x = x.unsqueeze(0).expand(y.shape[0], -1)

        # Compute sums
        sum_x = x.sum(dim=-1)
        sum_y = y.sum(dim=-1)
        sum_xy = (x * y).sum(dim=-1)
        sum_x2 = (x ** 2).sum(dim=-1)

        # Closed-form slope
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x2 - sum_x ** 2

        slope = numerator / (denominator + self.eps)

        return slope


class PinkNoiseLoss(SpectralSlopeLoss):
    """
    Convenience class specifically for pink noise (1/f) enforcement.

    Pink noise has a spectral slope of -1.0, meaning power decreases
    inversely with frequency. This is characteristic of many natural
    phenomena and high-quality music.
    """

    def __init__(self, **kwargs):
        super().__init__(target_slope=-1.0, **kwargs)


class BrownianNoiseLoss(SpectralSlopeLoss):
    """
    Convenience class for Brownian noise (1/f²) enforcement.

    Brownian noise has a spectral slope of -2.0, producing smoother,
    more predictable sequences.
    """

    def __init__(self, **kwargs):
        super().__init__(target_slope=-2.0, **kwargs)


class CombinedMusicLoss(nn.Module):
    """
    Combined loss for music generation with Pink Noise Prior.

    L_total = L_CE + λ * L_Pink

    This combines the standard cross-entropy loss for next-token prediction
    with the spectral slope loss for enforcing fractal structure.

    Args:
        lambda_pink: Weight for the pink noise loss (default: 0.1)
        target_slope: Target spectral slope (default: -1.0)
        vocab_size: Vocabulary size for computing expected pitch
        pitch_offset: Offset to convert token indices to pitch values
        label_smoothing: Label smoothing for cross-entropy loss
    """

    def __init__(
        self,
        lambda_pink: float = 0.1,
        target_slope: float = -1.0,
        vocab_size: int = 128,
        pitch_offset: int = 0,
        label_smoothing: float = 0.0,
        min_seq_len: int = 32,
    ):
        super().__init__()
        self.lambda_pink = lambda_pink
        self.vocab_size = vocab_size
        self.pitch_offset = pitch_offset
        self.min_seq_len = min_seq_len

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='mean',
        )
        self.spectral_loss = SpectralSlopeLoss(target_slope=target_slope)

        # Pre-compute pitch values for expected pitch calculation
        self.register_buffer(
            'pitch_values',
            torch.arange(vocab_size, dtype=torch.float32) + pitch_offset
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Compute the combined loss.

        Args:
            logits: Model output logits, shape (batch, seq_len, vocab_size)
            targets: Target token indices, shape (batch, seq_len)
            return_components: If True, return individual loss components

        Returns:
            total_loss: Combined loss value
            components: (optional) Dict with individual losses and metrics
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Cross-entropy loss
        # Reshape for cross_entropy: (batch * seq_len, vocab_size)
        ce_loss = self.ce_loss(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        # Spectral slope loss (on soft pitch sequence)
        if seq_len >= self.min_seq_len and self.lambda_pink > 0:
            # Convert logits to soft pitch sequence
            pitch_sequence = self._logits_to_soft_pitch(logits)
            spectral_loss, slope = self.spectral_loss(
                pitch_sequence, return_slope=True
            )
        else:
            spectral_loss = torch.tensor(0.0, device=logits.device)
            slope = torch.tensor(0.0, device=logits.device)

        # Combined loss
        total_loss = ce_loss + self.lambda_pink * spectral_loss

        if return_components:
            components = {
                'ce_loss': ce_loss.detach(),
                'spectral_loss': spectral_loss.detach(),
                'total_loss': total_loss.detach(),
                'spectral_slope': slope.detach(),
                'lambda_pink': self.lambda_pink,
            }
            return total_loss, components

        return total_loss

    def _logits_to_soft_pitch(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to a differentiable soft pitch sequence.

        Uses softmax probabilities as weights to compute expected pitch value
        at each position, maintaining differentiability.

        Args:
            logits: Shape (batch, seq_len, vocab_size)

        Returns:
            soft_pitch: Shape (batch, seq_len) - expected pitch at each position
        """
        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Expected pitch = sum(prob * pitch_value)
        # probs: (batch, seq_len, vocab_size)
        # pitch_values: (vocab_size,)
        soft_pitch = torch.einsum('bsv,v->bs', probs, self.pitch_values)

        return soft_pitch

    def set_lambda(self, lambda_pink: float):
        """Update the lambda weight for pink noise loss."""
        self.lambda_pink = lambda_pink


def generate_pink_noise(length: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate pink noise using the Voss-McCartney algorithm.

    Useful for testing that the spectral slope loss correctly identifies
    pink noise (loss should be near zero).

    Args:
        length: Length of the sequence to generate
        device: Torch device

    Returns:
        pink_noise: Tensor of shape (length,)
    """
    if device is None:
        device = torch.device('cpu')

    # Number of octaves (determines quality)
    n_octaves = int(math.log2(length)) + 1

    # Initialize
    pink = torch.zeros(length, device=device)

    # Voss-McCartney algorithm
    for octave in range(n_octaves):
        period = 2 ** octave
        # Random values that change at different rates
        values = torch.randn(length // period + 1, device=device)
        # Repeat to fill the sequence
        values = values.repeat_interleave(period)[:length]
        pink += values

    # Normalize
    pink = (pink - pink.mean()) / (pink.std() + 1e-8)

    return pink


def generate_white_noise(length: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate white noise (random, no correlation).

    Useful for testing that the spectral slope loss correctly identifies
    non-pink noise (loss should be high).

    Args:
        length: Length of the sequence to generate
        device: Torch device

    Returns:
        white_noise: Tensor of shape (length,)
    """
    if device is None:
        device = torch.device('cpu')
    return torch.randn(length, device=device)


def generate_brownian_noise(length: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate Brownian noise (cumulative sum of white noise).

    Args:
        length: Length of the sequence to generate
        device: Torch device

    Returns:
        brownian_noise: Tensor of shape (length,)
    """
    if device is None:
        device = torch.device('cpu')
    white = torch.randn(length, device=device)
    brownian = torch.cumsum(white, dim=0)
    # Normalize
    brownian = (brownian - brownian.mean()) / (brownian.std() + 1e-8)
    return brownian
