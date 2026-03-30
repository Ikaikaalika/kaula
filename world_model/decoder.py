"""Latent-to-frame decoder for rollout reconstruction.

Example:
    >>> import torch
    >>> from world_model.decoder import LatentDecoder
    >>> dec = LatentDecoder(out_channels=3, image_size=64, latent_dim=192)
    >>> frames = dec(torch.randn(2, 8, 192))
    >>> frames.shape
    torch.Size([2, 8, 3, 64, 64])
"""

from __future__ import annotations

import torch
from torch import nn


class LatentDecoder(nn.Module):
    """Decode latent states back into frames."""

    def __init__(self, out_channels: int, image_size: int, latent_dim: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.seed_hw = max(4, image_size // 8)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * self.seed_hw * self.seed_hw),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 4, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent states.

        Args:
            latents: Tensor with shape `[B, T, D]` or `[B, D]`.
        """
        squeeze_time = latents.ndim == 2
        if squeeze_time:
            latents = latents.unsqueeze(1)
        if latents.ndim != 3:
            raise ValueError(f"Expected latents rank 3 [B,T,D], got {tuple(latents.shape)}")

        bsz, steps, _ = latents.shape
        x = self.fc(latents.reshape(bsz * steps, self.latent_dim))
        x = x.reshape(bsz * steps, -1, self.seed_hw, self.seed_hw)
        y = self.net(x)
        if y.shape[-1] != self.image_size or y.shape[-2] != self.image_size:
            y = nn.functional.interpolate(y, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        y = y.reshape(bsz, steps, self.out_channels, self.image_size, self.image_size)
        if squeeze_time:
            return y[:, 0]
        return y
