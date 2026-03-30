"""BridgeData V2 dataset loader.

Expected layout (one common format):

    bridge_data_v2/
      trajectories/
        <episode_id>/
          images/*.jpg
          actions.npy (optional)

If actions are missing, zero actions are emitted with clear metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .common import collate_trajectory_batch, validate_required_path, _read_frame_dir_random_clip

BRIDGEDATA_LAYOUT = "bridge_data_v2/trajectories/<episode_id>/images/*.jpg (+ optional actions.npy)"


class BridgeDataDataset(Dataset):
    """Clip sampler for BridgeData V2 frame folders."""

    def __init__(
        self,
        episode_dirs: List[Path],
        clip_len: int,
        image_size: int,
        channels: int,
        action_dim: int,
        stride: int = 1,
        seed: int = 0,
        dataset_size: int = 10_000,
    ) -> None:
        self.episode_dirs = episode_dirs
        self.clip_len = clip_len
        self.image_size = image_size
        self.channels = channels
        self.action_dim = action_dim
        self.stride = stride
        self.seed = seed
        self.dataset_size = dataset_size

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> dict:
        import torch

        rng = np.random.default_rng(self.seed + idx)
        episode_dir = self.episode_dirs[int(rng.integers(0, len(self.episode_dirs)))]
        images_dir = episode_dir / "images"
        clip = _read_frame_dir_random_clip(
            frame_dir=images_dir,
            clip_len=self.clip_len,
            image_size=self.image_size,
            channels=self.channels,
            stride=self.stride,
            rng=rng,
        )

        actions_file = episode_dir / "actions.npy"
        if actions_file.exists():
            actions = np.load(actions_file)
            if actions.ndim != 2:
                raise ValueError(f"Expected actions.npy rank-2 [T, A], got shape {actions.shape}")
            needed = self.clip_len - 1
            if actions.shape[0] < needed:
                raise ValueError(
                    f"actions.npy in {episode_dir} has {actions.shape[0]} steps but requires {needed}."
                )
            if actions.shape[1] < self.action_dim:
                pad = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]), dtype=np.float32)
                actions = np.concatenate([actions.astype(np.float32), pad], axis=1)
            actions = actions[:needed, : self.action_dim].astype(np.float32)
            action_source = "file"
        else:
            actions = np.zeros((self.clip_len - 1, self.action_dim), dtype=np.float32)
            action_source = "zeros_missing_actions_npy"

        return {
            "frames": torch.from_numpy(clip.astype(np.float32)),
            "actions": torch.from_numpy(actions),
            "meta": {"source": str(episode_dir), "action_source": action_source},
        }


def _find_episode_dirs(root: Path) -> List[Path]:
    candidates = []
    for path in root.rglob("images"):
        if path.is_dir() and path.parent.is_dir():
            candidates.append(path.parent)
    return sorted(candidates)


def build_bridge_data_dataloader(
    data_root: str,
    batch_size: int,
    clip_len: int,
    image_size: int,
    action_dim: int,
    channels: int = 3,
    stride: int = 1,
    num_workers: int = 0,
    dataset_size: int = 10_000,
) -> DataLoader:
    """Build BridgeData dataloader."""
    root = validate_required_path(data_root, "BridgeData V2", BRIDGEDATA_LAYOUT)
    episodes = _find_episode_dirs(root)
    if not episodes:
        raise RuntimeError(
            f"No BridgeData episode directories found under {root}. Expected layout:\n{BRIDGEDATA_LAYOUT}"
        )

    ds = BridgeDataDataset(
        episode_dirs=episodes,
        clip_len=clip_len,
        image_size=image_size,
        channels=channels,
        action_dim=action_dim,
        stride=stride,
        dataset_size=dataset_size,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_trajectory_batch,
        drop_last=True,
    )
