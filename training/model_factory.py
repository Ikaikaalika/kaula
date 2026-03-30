"""Model factory shared by train/evaluate/rollout scripts."""

from __future__ import annotations

from typing import Dict

from world_model.decoder import LatentDecoder
from world_model.encoder import VisionEncoder
from world_model.latent_diffusion import LatentDiffusion
from world_model.ssm_diffusion_core import LatentStateSpaceDiffusionWorldModel, StateSpaceDiffusionCore


def build_model(cfg_model: Dict, cfg_diff: Dict) -> LatentStateSpaceDiffusionWorldModel:
    """Construct latent state-space diffusion world model from config dicts."""
    encoder = VisionEncoder(
        in_channels=int(cfg_model["channels"]),
        image_size=int(cfg_model["image_size"]),
        latent_dim=int(cfg_model["latent_dim"]),
        hidden_channels=int(cfg_model.get("encoder_hidden_channels", 64)),
    )
    core = StateSpaceDiffusionCore(
        latent_dim=int(cfg_model["latent_dim"]),
        action_dim=int(cfg_model["action_dim"]),
        hidden_dim=int(cfg_model["ssm_hidden_dim"]),
        backend=str(cfg_model.get("ssm_backend", "linear_recurrent")),
    )
    diffusion = LatentDiffusion(
        latent_dim=int(cfg_model["latent_dim"]),
        hidden_dim=int(cfg_diff.get("hidden_dim", cfg_model["ssm_hidden_dim"])),
        num_steps=int(cfg_diff["num_steps"]),
        schedule_name=str(cfg_diff.get("schedule", "cosine")),
        beta_start=float(cfg_diff.get("beta_start", 1e-4)),
        beta_end=float(cfg_diff.get("beta_end", 2e-2)),
        learned_variance=bool(cfg_diff.get("learned_variance", False)),
        time_dim=int(cfg_diff.get("time_dim", 128)),
    )
    decoder = LatentDecoder(
        out_channels=int(cfg_model["channels"]),
        image_size=int(cfg_model["image_size"]),
        latent_dim=int(cfg_model["latent_dim"]),
        hidden_channels=int(cfg_model.get("decoder_hidden_channels", 128)),
    )

    return LatentStateSpaceDiffusionWorldModel(
        encoder=encoder,
        core=core,
        diffusion=diffusion,
        decoder=decoder,
        backend_name=str(cfg_model.get("ssm_backend", "linear_recurrent")),
    )
