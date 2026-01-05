"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np
import torch

from pnp.metrics.fractal import (
    HiguchiFractalDimension,
    compute_fractal_dimension,
    compute_spectral_slope,
    analyze_sequence,
)
from pnp.metrics.zipf import ZipfAnalyzer, compute_zipf_coefficient
from pnp.metrics.autocorrelation import AutocorrelationAnalyzer, compute_autocorrelation


class TestHiguchiFractalDimension:
    """Tests for Higuchi Fractal Dimension."""

    def test_white_noise_fd(self):
        """White noise should have FD near 2.0."""
        hfd = HiguchiFractalDimension()

        white = np.random.randn(1024)
        fd, r2 = hfd(white)

        # White noise typically has FD around 1.5-2.0
        assert 1.4 < fd < 2.1, f"White noise FD {fd:.2f} outside expected range"

    def test_smooth_signal_fd(self):
        """Smooth signal should have low FD."""
        hfd = HiguchiFractalDimension()

        # Sine wave (very smooth)
        t = np.linspace(0, 10 * np.pi, 1024)
        smooth = np.sin(t)
        fd, r2 = hfd(smooth)

        # Smooth signals have FD near 1.0
        assert fd < 1.5, f"Smooth signal FD {fd:.2f} too high"

    def test_torch_input(self):
        """Should handle PyTorch tensors."""
        hfd = HiguchiFractalDimension()

        x = torch.randn(512)
        fd, r2 = hfd(x)

        assert not np.isnan(fd)
        assert 0 < r2 <= 1

    def test_short_sequence(self):
        """Should handle short sequences gracefully."""
        hfd = HiguchiFractalDimension()

        short = np.random.randn(20)
        fd, r2 = hfd(short)

        # May be nan for very short sequences
        assert np.isnan(fd) or (1.0 <= fd <= 2.0)


class TestSpectralSlope:
    """Tests for spectral slope computation."""

    def test_pink_noise_slope(self):
        """Pink noise should have slope near -1."""
        from pnp.losses.spectral import generate_pink_noise

        pink = generate_pink_noise(2048).numpy()
        slope, r2 = compute_spectral_slope(pink)

        assert -1.5 < slope < -0.5, f"Pink noise slope {slope:.2f} not near -1"

    def test_white_noise_slope(self):
        """White noise should have slope near 0."""
        white = np.random.randn(2048)
        slope, r2 = compute_spectral_slope(white)

        assert -0.5 < slope < 0.5, f"White noise slope {slope:.2f} not near 0"

    def test_brownian_noise_slope(self):
        """Brownian noise should have slope near -2."""
        from pnp.losses.spectral import generate_brownian_noise

        brownian = generate_brownian_noise(2048).numpy()
        slope, r2 = compute_spectral_slope(brownian)

        assert slope < -1.0, f"Brownian noise slope {slope:.2f} not steep enough"


class TestAnalyzeSequence:
    """Tests for complete sequence analysis."""

    def test_complete_analysis(self):
        """analyze_sequence should return all metrics."""
        x = np.random.randn(1024)
        result = analyze_sequence(x)

        assert hasattr(result, 'fractal_dimension')
        assert hasattr(result, 'spectral_slope')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'is_valid')


class TestZipfAnalyzer:
    """Tests for Zipf's law analysis."""

    def test_zipfian_distribution(self):
        """Zipf-like distribution should have coefficient near 1."""
        # Generate Zipf-like data
        # Frequency proportional to 1/rank
        elements = []
        for rank in range(1, 51):
            freq = int(1000 / rank)
            elements.extend([rank] * freq)
        np.random.shuffle(elements)

        analyzer = ZipfAnalyzer()
        result = analyzer(elements)

        assert 0.5 < result.coefficient < 1.5, \
            f"Zipf coefficient {result.coefficient:.2f} outside expected range"

    def test_uniform_distribution(self):
        """Uniform distribution should have low coefficient."""
        # Uniform - each element has same frequency
        elements = list(range(100)) * 10  # 10 of each
        np.random.shuffle(elements)

        analyzer = ZipfAnalyzer()
        result = analyzer(elements)

        # Uniform distribution has coefficient near 0
        assert result.coefficient < 0.5, \
            f"Uniform distribution coefficient {result.coefficient:.2f} too high"

    def test_list_input(self):
        """Should handle list input."""
        elements = [1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        result = compute_zipf_coefficient(elements)
        assert isinstance(result, float)

    def test_tensor_input(self):
        """Should handle tensor input."""
        elements = torch.randint(0, 50, (500,))

        analyzer = ZipfAnalyzer()
        result = analyzer(elements)

        assert result.unique_elements > 0


class TestAutocorrelationAnalyzer:
    """Tests for autocorrelation analysis."""

    def test_white_noise_acf(self):
        """White noise should have fast-decaying ACF."""
        analyzer = AutocorrelationAnalyzer()

        white = np.random.randn(1024)
        result = analyzer(white)

        # White noise has rapid decay
        assert result.decay_rate > 0.1, \
            f"White noise decay rate {result.decay_rate:.4f} too slow"
        # Low long-range correlation
        assert abs(result.long_range_correlation) < 0.2

    def test_structured_signal_acf(self):
        """Structured signal should have slower-decaying ACF."""
        analyzer = AutocorrelationAnalyzer()

        # Create signal with long-range structure
        t = np.linspace(0, 20 * np.pi, 1024)
        structured = np.sin(t) + 0.5 * np.sin(2 * t) + np.random.randn(1024) * 0.1
        result = analyzer(structured)

        # Should have significant correlations
        assert result.long_range_correlation > 0.1 or result.characteristic_lag > 10

    def test_acf_values(self):
        """ACF values should be returned."""
        analyzer = AutocorrelationAnalyzer()

        x = np.random.randn(512)
        result = analyzer(x)

        assert len(result.acf_values) > 0
        # ACF at lag 0 should be 1
        assert abs(result.acf_values[0] - 1.0) < 0.01

    def test_tensor_input(self):
        """Should handle tensor input."""
        x = torch.randn(512)
        acf = compute_autocorrelation(x)

        assert isinstance(acf, np.ndarray)
        assert len(acf) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
