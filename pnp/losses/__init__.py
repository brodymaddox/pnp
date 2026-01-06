"""Loss functions for Pink Noise Prior music generation."""

from pnp.losses.spectral import SpectralSlopeLoss, PinkNoiseLoss, CombinedMusicLoss

__all__ = ["SpectralSlopeLoss", "PinkNoiseLoss", "CombinedMusicLoss"]
