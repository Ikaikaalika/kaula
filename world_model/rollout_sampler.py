"""Rollout sampling helpers for trajectory generation.

Example:
    >>> batch = next(iter(loader))
    >>> out = sample_rollout(model, batch["frames"][:, :4], batch["actions"], horizon=8, num_samples=4)
"""

from __future__ import annotations

from typing import Dict

import torch

from .ssm_diffusion_core import LatentStateSpaceDiffusionWorldModel


@torch.no_grad()
def sample_rollout(
    model: LatentStateSpaceDiffusionWorldModel,
    context_frames: torch.Tensor,
    action_sequence: torch.Tensor,
    horizon: int,
    num_samples: int,
) -> Dict[str, torch.Tensor]:
    """Wrapper for stochastic rollout sampling.

    Args:
        model: Latent state-space diffusion world model.
        context_frames: `[B, T_ctx, C, H, W]` observed frames.
        action_sequence: `[B, horizon, A]` actions for forecast horizon.
        horizon: Number of steps to predict.
        num_samples: Number of sampled trajectories.
    """
    return model.rollout(
        context_frames=context_frames,
        action_sequence=action_sequence,
        rollout_horizon=horizon,
        num_samples=num_samples,
    )
