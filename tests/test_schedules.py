"""Unit tests for diffusion schedules."""

from __future__ import annotations

import torch

from world_model.schedules import build_diffusion_schedule, extract


def test_build_cosine_schedule_shapes() -> None:
    schedule = build_diffusion_schedule("cosine", num_steps=16)
    assert schedule["betas"].shape == (16,)
    assert schedule["alphas"].shape == (16,)
    assert schedule["alpha_bars"].shape == (16,)


def test_extract_broadcast_shape() -> None:
    schedule = build_diffusion_schedule("linear", num_steps=8)
    t = torch.tensor([0, 2, 7], dtype=torch.long)
    out = extract(schedule["alpha_bars"], t, shape=(3, 5))
    assert out.shape == (3, 1)
