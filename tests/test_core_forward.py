"""Unit tests for integrated state-space diffusion core."""

from __future__ import annotations

import torch

from world_model.decoder import LatentDecoder
from world_model.encoder import VisionEncoder
from world_model.latent_diffusion import LatentDiffusion
from world_model.ssm_diffusion_core import LatentStateSpaceDiffusionWorldModel, StateSpaceDiffusionCore


def _build_model() -> LatentStateSpaceDiffusionWorldModel:
    encoder = VisionEncoder(in_channels=3, image_size=32, latent_dim=64, hidden_channels=32)
    core = StateSpaceDiffusionCore(latent_dim=64, action_dim=4, hidden_dim=96, backend="linear_recurrent")
    diffusion = LatentDiffusion(latent_dim=64, hidden_dim=96, num_steps=8, schedule_name="linear")
    decoder = LatentDecoder(out_channels=3, image_size=32, latent_dim=64, hidden_channels=64)
    return LatentStateSpaceDiffusionWorldModel(
        encoder=encoder,
        core=core,
        diffusion=diffusion,
        decoder=decoder,
        backend_name="linear_recurrent",
    )


def test_forward_train_shapes() -> None:
    model = _build_model()
    frames = torch.randn(2, 8, 3, 32, 32)
    actions = torch.randn(2, 7, 4)
    out = model.forward_train(frames, actions)
    assert out["pred_latents"].shape == (2, 7, 64)
    assert out["decoded_frames"].shape == (2, 7, 3, 32, 32)


def test_rollout_shapes() -> None:
    model = _build_model()
    frames = torch.randn(2, 4, 3, 32, 32)
    actions = torch.randn(2, 8, 4)
    out = model.rollout(context_frames=frames, action_sequence=actions, rollout_horizon=8, num_samples=3)
    assert out["latent_trajectory"].shape == (2, 3, 8, 64)
    assert out["decoded_trajectory"].shape == (2, 3, 8, 3, 32, 32)
    assert out["uncertainty"].shape == (2, 8, 64)
