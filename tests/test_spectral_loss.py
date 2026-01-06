"""
Tests for the Spectral Slope Loss function.

These tests verify that:
1. The loss correctly identifies pink noise (slope ≈ -1)
2. The loss correctly penalizes white noise (slope ≈ 0)
3. The loss correctly penalizes brownian noise (slope ≈ -2)
4. Gradients flow properly through the loss
"""

import pytest
import torch
import numpy as np

from pnp.losses.spectral import (
    SpectralSlopeLoss,
    PinkNoiseLoss,
    CombinedMusicLoss,
    generate_pink_noise,
    generate_white_noise,
    generate_brownian_noise,
)


class TestSpectralSlopeLoss:
    """Tests for SpectralSlopeLoss."""

    def test_pink_noise_low_loss(self):
        """Pink noise should have low loss (slope near -1)."""
        loss_fn = SpectralSlopeLoss(target_slope=-1.0)

        # Generate pink noise
        pink = generate_pink_noise(1024)

        loss, slope = loss_fn(pink, return_slope=True)

        # Pink noise should have slope near -1
        assert abs(slope.item() - (-1.0)) < 0.5, \
            f"Pink noise slope {slope.item():.2f} not near -1"
        # Loss should be relatively low
        assert loss.item() < 0.5, \
            f"Pink noise loss {loss.item():.4f} unexpectedly high"

    def test_white_noise_high_loss(self):
        """White noise should have high loss (slope near 0)."""
        loss_fn = SpectralSlopeLoss(target_slope=-1.0)

        # Generate white noise
        white = generate_white_noise(1024)

        loss, slope = loss_fn(white, return_slope=True)

        # White noise should have slope near 0
        assert abs(slope.item()) < 0.5, \
            f"White noise slope {slope.item():.2f} not near 0"
        # Loss should be high (targeting -1 but getting ~0)
        assert loss.item() > 0.3, \
            f"White noise loss {loss.item():.4f} unexpectedly low"

    def test_brownian_noise_loss(self):
        """Brownian noise should have slope near -2."""
        loss_fn = SpectralSlopeLoss(target_slope=-2.0)

        # Generate brownian noise
        brownian = generate_brownian_noise(1024)

        loss, slope = loss_fn(brownian, return_slope=True)

        # Brownian noise should have slope near -2
        assert slope.item() < -1.0, \
            f"Brownian noise slope {slope.item():.2f} not < -1"

    def test_gradient_flow(self):
        """Gradients should flow through the loss."""
        loss_fn = SpectralSlopeLoss(target_slope=-1.0)

        # Create differentiable input
        x = torch.randn(256, requires_grad=True)

        loss = loss_fn(x)
        loss.backward()

        # Check gradients exist and are finite
        assert x.grad is not None, "No gradients computed"
        assert torch.isfinite(x.grad).all(), "Gradients contain inf/nan"
        assert (x.grad != 0).any(), "All gradients are zero"

    def test_batch_processing(self):
        """Loss should handle batched inputs."""
        loss_fn = SpectralSlopeLoss(target_slope=-1.0)

        # Batch of sequences
        batch = torch.stack([
            generate_pink_noise(256),
            generate_white_noise(256),
            generate_brownian_noise(256),
        ])

        loss = loss_fn(batch)

        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_short_sequence_handling(self):
        """Loss should handle short sequences gracefully."""
        loss_fn = SpectralSlopeLoss(target_slope=-1.0)

        # Very short sequence
        short = torch.randn(10)
        loss = loss_fn(short)

        # Should not raise error
        assert torch.isfinite(loss) or loss.item() == 0.0


class TestPinkNoiseLoss:
    """Tests for PinkNoiseLoss convenience class."""

    def test_target_slope(self):
        """PinkNoiseLoss should target slope -1."""
        loss_fn = PinkNoiseLoss()

        assert loss_fn.target_slope == -1.0

    def test_low_loss_for_pink_noise(self):
        """PinkNoiseLoss should give low loss for pink noise."""
        loss_fn = PinkNoiseLoss()

        pink = generate_pink_noise(1024)
        loss = loss_fn(pink)

        assert loss.item() < 0.5


class TestCombinedMusicLoss:
    """Tests for CombinedMusicLoss."""

    def test_components(self):
        """Combined loss should return both CE and spectral components."""
        vocab_size = 128
        loss_fn = CombinedMusicLoss(
            lambda_pink=0.1,
            target_slope=-1.0,
            vocab_size=vocab_size,
        )

        # Create dummy logits and targets
        batch_size, seq_len = 4, 64
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        total_loss, components = loss_fn(logits, targets, return_components=True)

        # Check all components present
        assert 'ce_loss' in components
        assert 'spectral_loss' in components
        assert 'total_loss' in components
        assert 'spectral_slope' in components

        # Check total is combination
        expected = components['ce_loss'] + 0.1 * components['spectral_loss']
        assert torch.isclose(total_loss, expected, atol=1e-5)

    def test_lambda_zero_equals_ce_only(self):
        """With lambda=0, loss should equal CE loss."""
        vocab_size = 128
        loss_fn = CombinedMusicLoss(
            lambda_pink=0.0,
            vocab_size=vocab_size,
        )

        batch_size, seq_len = 4, 64
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        total_loss, components = loss_fn(logits, targets, return_components=True)

        # Total should equal CE loss
        assert torch.isclose(total_loss, components['ce_loss'], atol=1e-6)

    def test_soft_pitch_differentiable(self):
        """Soft pitch conversion should be differentiable."""
        vocab_size = 128
        loss_fn = CombinedMusicLoss(
            lambda_pink=0.1,
            vocab_size=vocab_size,
        )

        batch_size, seq_len = 2, 64
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_set_lambda(self):
        """set_lambda should update the weight."""
        loss_fn = CombinedMusicLoss(lambda_pink=0.1)

        loss_fn.set_lambda(0.5)
        assert loss_fn.lambda_pink == 0.5

        loss_fn.set_lambda(0.0)
        assert loss_fn.lambda_pink == 0.0


class TestNoiseGenerators:
    """Tests for noise generation utilities."""

    def test_pink_noise_shape(self):
        """Pink noise should have correct shape."""
        noise = generate_pink_noise(1024)
        assert noise.shape == (1024,)

    def test_white_noise_shape(self):
        """White noise should have correct shape."""
        noise = generate_white_noise(1024)
        assert noise.shape == (1024,)

    def test_brownian_noise_shape(self):
        """Brownian noise should have correct shape."""
        noise = generate_brownian_noise(1024)
        assert noise.shape == (1024,)

    def test_normalized_output(self):
        """All noise should be approximately normalized."""
        for gen_fn in [generate_pink_noise, generate_white_noise, generate_brownian_noise]:
            noise = gen_fn(1024)

            # Should have mean near 0 and std near 1
            assert abs(noise.mean().item()) < 0.2, f"{gen_fn.__name__} mean not near 0"
            assert abs(noise.std().item() - 1.0) < 0.3, f"{gen_fn.__name__} std not near 1"

    def test_device_placement(self):
        """Noise generators should respect device argument."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            noise = generate_pink_noise(256, device=device)
            assert noise.device.type == 'cuda'


class TestSpectralAnalysis:
    """Additional spectral analysis tests."""

    def test_spectral_slope_computation(self):
        """Test spectral slope estimation accuracy."""
        from pnp.metrics.fractal import compute_spectral_slope

        # Create signal with known characteristics
        pink = generate_pink_noise(2048).numpy()
        slope, r2 = compute_spectral_slope(pink)

        # Pink noise should have slope near -1
        assert -1.5 < slope < -0.5, f"Slope {slope:.2f} outside expected range"
        # Fit should be reasonable
        assert r2 > 0.5, f"R² {r2:.2f} too low"

    def test_different_sequence_lengths(self):
        """Loss should work with various sequence lengths."""
        loss_fn = SpectralSlopeLoss(target_slope=-1.0)

        for length in [64, 128, 256, 512, 1024]:
            pink = generate_pink_noise(length)
            loss = loss_fn(pink)

            assert torch.isfinite(loss), f"Loss not finite for length {length}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
