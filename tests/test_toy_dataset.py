"""Unit tests for toy dataset loader."""

from __future__ import annotations

from datasets.toy_dataset import build_toy_dataloader


def test_toy_loader_shapes() -> None:
    loader = build_toy_dataloader(
        batch_size=2,
        clip_len=8,
        image_size=32,
        action_dim=4,
        channels=3,
        num_samples=16,
    )
    batch = next(iter(loader))
    assert batch["frames"].shape == (2, 8, 3, 32, 32)
    assert batch["actions"].shape == (2, 7, 4)
