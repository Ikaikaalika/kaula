"""Loss stack for latent state-space diffusion world model.

Total loss:
    L = lambda1 diffusion + lambda2 latent_rollout + lambda3 reconstruction + lambda4 distillation + lambda5 contrastive

Example:
    >>> from world_model.losses import LossWeights, compute_world_model_losses
    >>> weights = LossWeights()
    >>> losses = compute_world_model_losses(outputs, weights)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Training weights for world model objective terms."""

    diffusion: float = 1.0
    latent_rollout: float = 1.0
    reconstruction: float = 1.0
    distillation: float = 0.0
    contrastive_alignment: float = 0.0


def _cosine_alignment(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return 1.0 - (a_n * b_n).sum(dim=-1).mean()


def compute_world_model_losses(outputs: Dict[str, torch.Tensor], weights: LossWeights) -> Dict[str, torch.Tensor]:
    """Compute weighted training losses.

    Required output keys:
    - `pred_noise`, `target_noise`
    - `pred_latents`, `target_latents`
    - `decoded_frames`, `target_frames`

    Optional output keys:
    - `distill_pred`, `distill_target`
    - `align_a`, `align_b`
    """
    diffusion_loss = F.mse_loss(outputs["pred_noise"], outputs["target_noise"])
    latent_rollout_loss = F.mse_loss(outputs["pred_latents"], outputs["target_latents"])
    reconstruction_loss = F.mse_loss(outputs["decoded_frames"], outputs["target_frames"])

    if "distill_pred" in outputs and "distill_target" in outputs:
        distillation_loss = F.mse_loss(outputs["distill_pred"], outputs["distill_target"])
    else:
        distillation_loss = torch.zeros((), device=diffusion_loss.device)

    if "align_a" in outputs and "align_b" in outputs:
        contrastive_alignment_loss = _cosine_alignment(outputs["align_a"], outputs["align_b"])
    else:
        contrastive_alignment_loss = torch.zeros((), device=diffusion_loss.device)

    total = (
        weights.diffusion * diffusion_loss
        + weights.latent_rollout * latent_rollout_loss
        + weights.reconstruction * reconstruction_loss
        + weights.distillation * distillation_loss
        + weights.contrastive_alignment * contrastive_alignment_loss
    )

    return {
        "loss": total,
        "diffusion_loss": diffusion_loss,
        "latent_rollout_loss": latent_rollout_loss,
        "reconstruction_loss": reconstruction_loss,
        "distillation_loss": distillation_loss,
        "contrastive_alignment_loss": contrastive_alignment_loss,
    }
