"""Latent diffusion primitives used inside state-space temporal dynamics.

This module intentionally models diffusion in latent state space only.

Example:
    >>> import torch
    >>> from world_model.latent_diffusion import LatentDiffusion
    >>> diffusion = LatentDiffusion(latent_dim=128, hidden_dim=256, num_steps=32)
    >>> x0 = torch.randn(4, 128)
    >>> t = torch.randint(0, 32, (4,))
    >>> noise = torch.randn_like(x0)
    >>> xt = diffusion.q_sample(x0, t, noise)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .schedules import build_diffusion_schedule, extract


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must be [B], got {tuple(timesteps.shape)}")
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class LatentDenoiser(nn.Module):
    """Reverse diffusion network conditioned on SSM hidden state."""

    def __init__(self, latent_dim: int, hidden_dim: int, time_dim: int, learned_variance: bool = False) -> None:
        super().__init__()
        self.learned_variance = learned_variance
        out_dim = latent_dim * (2 if learned_variance else 1)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + latent_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        conditioning_latent: torch.Tensor,
        hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict diffusion noise (and optional learned variance)."""
        t_embed = self.time_embed(timesteps)
        x = torch.cat([noisy_latent, conditioning_latent, hidden_state, t_embed], dim=-1)
        out = self.net(x)
        if not self.learned_variance:
            return out, None
        pred_noise, log_var = torch.chunk(out, chunks=2, dim=-1)
        return pred_noise, log_var


class LatentDiffusion(nn.Module):
    """Forward and reverse latent diffusion utilities."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_steps: int,
        schedule_name: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        learned_variance: bool = False,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        schedule = build_diffusion_schedule(
            schedule_name=schedule_name,
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.num_steps = num_steps
        self.learned_variance = learned_variance
        self.register_buffer("betas", schedule["betas"], persistent=False)
        self.register_buffer("alphas", schedule["alphas"], persistent=False)
        self.register_buffer("alpha_bars", schedule["alpha_bars"], persistent=False)
        self.register_buffer("alpha_bars_prev", schedule["alpha_bars_prev"], persistent=False)

        self.denoiser = LatentDenoiser(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            learned_variance=learned_variance,
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, clean_latent: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward noising: q(z^k | z^0)."""
        if noise is None:
            noise = torch.randn_like(clean_latent)
        alpha_bar_t = extract(self.alpha_bars, timesteps, clean_latent.shape)
        return torch.sqrt(alpha_bar_t) * clean_latent + torch.sqrt(1.0 - alpha_bar_t) * noise

    def predict_start_from_noise(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover z^0 estimate from noisy latent and predicted noise."""
        alpha_bar_t = extract(self.alpha_bars, timesteps, noisy_latent.shape)
        return (noisy_latent - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t.clamp_min(1e-8))

    def reverse_sample(
        self,
        hidden_state: torch.Tensor,
        conditioning_latent: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Sample z_{t+1} via reverse diffusion conditioned on SSM state.

        Returns:
            Tensor with shape `[B, S, D]`.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        bsz, latent_dim = conditioning_latent.shape
        out = []
        for _ in range(num_samples):
            x = torch.randn(bsz, latent_dim, device=conditioning_latent.device)
            for k in reversed(range(self.num_steps)):
                t = torch.full((bsz,), k, device=conditioning_latent.device, dtype=torch.long)
                pred_eps, _ = self.denoiser(x, conditioning_latent, hidden_state, t)
                x0 = self.predict_start_from_noise(x, t, pred_eps)
                if k == 0:
                    x = x0
                    continue
                alpha_bar_prev = extract(self.alpha_bars_prev, t, x.shape)
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * noise
            out.append(x)
        return torch.stack(out, dim=1)

    def summary(self) -> Dict[str, float]:
        """Return schedule metadata for logging."""
        return {
            "num_steps": float(self.num_steps),
            "beta_min": float(self.betas.min().item()),
            "beta_max": float(self.betas.max().item()),
        }
