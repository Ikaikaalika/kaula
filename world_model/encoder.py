"""Vision encoder that maps frame clips into latent state sequences.

Example:
    >>> import torch
    >>> from world_model.encoder import VisionEncoder
    >>> enc = VisionEncoder(in_channels=3, image_size=64, latent_dim=192)
    >>> frames = torch.randn(2, 8, 3, 64, 64)
    >>> latents = enc(frames)
    >>> latents.shape
    torch.Size([2, 8, 192])
"""

from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """Convolutional encoder for `[B, T, C, H, W]` video clips."""

    def __init__(self, in_channels: int, image_size: int, latent_dim: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(hidden_channels * 4, latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames into latent states.

        Args:
            frames: Tensor with shape `[B, T, C, H, W]`.
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected frames with rank 5 [B,T,C,H,W], got {tuple(frames.shape)}")
        bsz, steps, channels, height, width = frames.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"VisionEncoder expected {self.image_size}x{self.image_size} frames, got {height}x{width}"
            )
        x = frames.reshape(bsz * steps, channels, height, width)
        h = self.net(x).flatten(1)
        z = self.proj(h)
        return z.reshape(bsz, steps, self.latent_dim)
