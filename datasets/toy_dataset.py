"""Toy synthetic dataset for smoke tests and local debugging.

Example:
    >>> from datasets.toy_dataset import build_toy_dataloader
    >>> loader = build_toy_dataloader(batch_size=4, clip_len=8, image_size=64, action_dim=4)
    >>> batch = next(iter(loader))
    >>> batch["frames"].shape
    torch.Size([4, 8, 3, 64, 64])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import collate_trajectory_batch


@dataclass
class ToyDatasetConfig:
    """Configuration for toy synthetic trajectories."""

    clip_len: int = 8
    image_size: int = 64
    channels: int = 3
    action_dim: int = 4
    num_samples: int = 2048
    seed: int = 7


class ToyTrajectoryDataset(Dataset):
    """Moving shape trajectories with deterministic per-index generation."""

    def __init__(self, cfg: ToyDatasetConfig) -> None:
        self.cfg = cfg

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.cfg.seed + idx)
        clip_len = self.cfg.clip_len
        image_size = self.cfg.image_size
        channels = self.cfg.channels

        frames = np.zeros((clip_len, channels, image_size, image_size), dtype=np.float32)
        actions = np.zeros((clip_len - 1, self.cfg.action_dim), dtype=np.float32)

        cx = int(rng.integers(10, image_size - 10))
        cy = int(rng.integers(10, image_size - 10))
        vx = int(rng.choice([-2, -1, 1, 2]))
        vy = int(rng.choice([-2, -1, 1, 2]))
        shape = int(rng.choice([0, 1]))  # 0 square, 1 circle

        yy, xx = np.mgrid[0:image_size, 0:image_size]

        for t in range(clip_len):
            canvas = np.zeros((image_size, image_size), dtype=np.float32)
            if shape == 0:
                x0, x1 = max(0, cx - 4), min(image_size, cx + 4)
                y0, y1 = max(0, cy - 4), min(image_size, cy + 4)
                canvas[y0:y1, x0:x1] = 1.0
            else:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 16
                canvas[mask] = 1.0

            for c in range(channels):
                frames[t, c] = canvas

            if t < clip_len - 1:
                bounce = 0.0
                nx, ny = cx + vx, cy + vy
                if nx < 6 or nx > image_size - 6:
                    vx *= -1
                    bounce = 1.0
                if ny < 6 or ny > image_size - 6:
                    vy *= -1
                    bounce = 1.0
                actions[t] = np.array([
                    vx / 2.0,
                    vy / 2.0,
                    float(shape),
                    bounce,
                ], dtype=np.float32)

            cx = int(np.clip(cx + vx, 6, image_size - 6))
            cy = int(np.clip(cy + vy, 6, image_size - 6))

        return {
            "frames": torch.from_numpy(frames),
            "actions": torch.from_numpy(actions),
            "meta": {"dataset": "toy", "index": idx},
        }


def build_toy_dataloader(
    batch_size: int,
    clip_len: int,
    image_size: int,
    action_dim: int,
    channels: int = 3,
    num_samples: int = 2048,
    seed: int = 7,
    num_workers: int = 0,
) -> DataLoader:
    """Construct toy dataloader.

    Args:
        batch_size: Number of trajectories per batch.
        clip_len: Clip length T.
        image_size: Square frame size.
        action_dim: Action feature size.
    """
    cfg = ToyDatasetConfig(
        clip_len=clip_len,
        image_size=image_size,
        channels=channels,
        action_dim=action_dim,
        num_samples=num_samples,
        seed=seed,
    )
    ds = ToyTrajectoryDataset(cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_trajectory_batch,
        drop_last=True,
    )
