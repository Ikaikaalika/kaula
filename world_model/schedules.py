"""Diffusion schedules for latent-space state-space dynamics.

Example:
    >>> import torch
    >>> from world_model.schedules import build_diffusion_schedule, extract
    >>> schedule = build_diffusion_schedule("cosine", num_steps=8)
    >>> t = torch.tensor([0, 3, 7])
    >>> alpha_bar_t = extract(schedule["alpha_bars"], t, shape=(3, 64))
"""

from __future__ import annotations

import math
from typing import Dict

import torch


ScheduleDict = Dict[str, torch.Tensor]


def _cosine_alpha_bar(step: int, num_steps: int, s: float = 0.008) -> float:
    """Compute cosine cumulative alpha bar at a given diffusion step."""
    t = step / num_steps
    return math.cos((t + s) / (1.0 + s) * math.pi * 0.5) ** 2


def _build_cosine_betas(num_steps: int) -> torch.Tensor:
    alpha_bars = []
    for step in range(num_steps + 1):
        alpha_bars.append(_cosine_alpha_bar(step, num_steps))
    betas = []
    for step in range(1, num_steps + 1):
        beta = min(1.0 - (alpha_bars[step] / alpha_bars[step - 1]), 0.999)
        betas.append(beta)
    return torch.tensor(betas, dtype=torch.float32)


def _build_linear_betas(num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps=num_steps, dtype=torch.float32)


def build_diffusion_schedule(
    schedule_name: str,
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> ScheduleDict:
    """Create diffusion schedule tensors.

    Args:
        schedule_name: One of `"cosine"` or `"linear"`.
        num_steps: Number of diffusion steps K.
        beta_start: Start beta for linear schedule.
        beta_end: End beta for linear schedule.

    Returns:
        Dict containing betas, alphas, alpha_bars, and alpha_bars_prev.
    """
    name = schedule_name.lower().strip()
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    if name == "cosine":
        betas = _build_cosine_betas(num_steps)
    elif name == "linear":
        betas = _build_linear_betas(num_steps, beta_start=beta_start, beta_end=beta_end)
    else:
        raise ValueError(f"Unsupported diffusion schedule '{schedule_name}'. Expected 'cosine' or 'linear'.")

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
    }


def extract(schedule_tensor: torch.Tensor, timesteps: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Gather schedule values by timestep and broadcast to requested shape.

    Args:
        schedule_tensor: Tensor indexed by diffusion step, shape [K].
        timesteps: Integer timestep tensor, shape [B].
        shape: Output target shape with leading batch size B.
    """
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be rank-1 [B], got shape {tuple(timesteps.shape)}")
    gathered = schedule_tensor.gather(0, timesteps)
    while gathered.ndim < len(shape):
        gathered = gathered.unsqueeze(-1)
    return gathered
