from __future__ import annotations

import mlx.core as mx

from .config import WorldModelConfig


def mse(x, y):
    return mx.mean(mx.square(x - y))


def world_model_loss(cfg: WorldModelConfig, outputs, frames, noise):
    latent_targets = outputs["pooled_latents"][:, 1:]
    latent_loss = mse(outputs["z_preds"], latent_targets)
    recon_loss = mse(outputs["next_frame_pred"], frames[:, -1])
    diff_loss = mse(outputs["eps_hat"], noise)
    total = (
        cfg.latent_weight * latent_loss
        + cfg.recon_weight * recon_loss
        + cfg.diffusion_weight * diff_loss
    )
    return total, {
        "loss": total,
        "latent_loss": latent_loss,
        "recon_loss": recon_loss,
        "diffusion_loss": diff_loss,
    }
