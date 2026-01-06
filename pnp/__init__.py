"""
Pink Noise Prior: Enforcing Fractal Dynamics in Symbolic Music Generation
via Differentiable Spectral Regularization

This package implements a novel approach to music generation by incorporating
a differentiable spectral slope loss that enforces 1/f (pink noise) characteristics
during training, promoting long-range structural coherence in generated music.
"""

__version__ = "0.1.0"
__author__ = "Research Project"

from pnp.losses.spectral import SpectralSlopeLoss, PinkNoiseLoss
from pnp.models.transformer import MusicTransformer
from pnp.metrics.fractal import HiguchiFractalDimension, compute_spectral_slope
from pnp.metrics.zipf import ZipfAnalyzer
from pnp.metrics.autocorrelation import AutocorrelationAnalyzer

__all__ = [
    "SpectralSlopeLoss",
    "PinkNoiseLoss",
    "MusicTransformer",
    "HiguchiFractalDimension",
    "compute_spectral_slope",
    "ZipfAnalyzer",
    "AutocorrelationAnalyzer",
]
