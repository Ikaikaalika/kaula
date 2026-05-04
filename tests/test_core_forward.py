"""Unit tests for integrated state-space diffusion core."""

from __future__ import annotations

import torch

from world_model.decoder import LatentDecoder
from world_model.encoder import VisionEncoder
from world_model.latent_diffusion import LatentDiffusion
from world_model.losses import LossWeights, compute_world_model_losses
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
    assert out["latent_prior"].shape == (2, 7, 64)
    assert out["target_latents"].shape == (2, 7, 64)
    assert out["pred_noise"].shape == (2, 7, 64)
    assert out["target_noise"].shape == (2, 7, 64)


def test_forward_train_scheduled_sampling_runs() -> None:
    model = _build_model()
    frames = torch.randn(2, 6, 3, 32, 32)
    actions = torch.randn(2, 5, 4)
    for tf in (0.0, 0.5, 1.0):
        out = model.forward_train(frames, actions, teacher_forcing_prob=tf)
        assert out["pred_latents"].shape == (2, 5, 64)
        assert torch.isfinite(out["pred_latents"]).all()


def test_forward_train_loss_backprop() -> None:
    model = _build_model()
    frames = torch.randn(2, 6, 3, 32, 32)
    actions = torch.randn(2, 5, 4)
    out = model.forward_train(frames, actions, teacher_forcing_prob=0.5)
    losses = compute_world_model_losses(out, LossWeights())
    losses["loss"].backward()
    grads_seen = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grads_seen > 0
    assert "latent_prior_loss" in losses


def test_rollout_shapes() -> None:
    model = _build_model()
    frames = torch.randn(2, 4, 3, 32, 32)
    actions = torch.randn(2, 8, 4)
    out = model.rollout(context_frames=frames, action_sequence=actions, rollout_horizon=8, num_samples=3)
    assert out["latent_trajectory"].shape == (2, 3, 8, 64)
    assert out["decoded_trajectory"].shape == (2, 3, 8, 3, 32, 32)
    assert out["uncertainty"].shape == (2, 8, 64)


def test_rollout_determinism_with_fixed_seed() -> None:
    model = _build_model()
    model.eval()
    frames = torch.randn(2, 4, 3, 32, 32)
    actions = torch.randn(2, 8, 4)

    torch.manual_seed(42)
    out_a = model.rollout(context_frames=frames, action_sequence=actions, rollout_horizon=8, num_samples=2)
    torch.manual_seed(42)
    out_b = model.rollout(context_frames=frames, action_sequence=actions, rollout_horizon=8, num_samples=2)

    assert torch.allclose(out_a["latent_trajectory"], out_b["latent_trajectory"])
    assert torch.allclose(out_a["uncertainty"], out_b["uncertainty"])


def test_rollout_uncertainty_is_nonzero_with_multiple_samples() -> None:
    model = _build_model()
    model.eval()
    frames = torch.randn(2, 4, 3, 32, 32)
    actions = torch.randn(2, 8, 4)
    out = model.rollout(context_frames=frames, action_sequence=actions, rollout_horizon=8, num_samples=4)
    assert out["uncertainty"].max() > 0, "stochastic sampling should produce nonzero uncertainty"
