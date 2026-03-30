"""DROID RLDS dataset loader.

Expected layout (TensorFlow Datasets root):

    <data_root>/droid/<version>/

Requires optional dependencies:
    tensorflow
    tensorflow-datasets

When dependencies are unavailable, loader fails with a clear message.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import _normalize_frame, collate_trajectory_batch, validate_required_path

DROID_LAYOUT = "<tfds_root>/droid/<version>/ (TFDS RLDS format)"


def _optional_import_tfds():
    try:
        import tensorflow_datasets as tfds
    except Exception as exc:
        raise ImportError(
            "DROID loader requires tensorflow_datasets and tensorflow. Install requirements-datasets.txt."
        ) from exc
    return tfds


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _find_first_image(obj: Any):
    preferred = [
        "exterior_image_1_left",
        "exterior_image_2_left",
        "wrist_image_left",
        "wrist_image_right",
        "image",
        "rgb",
        "pixels",
    ]
    if isinstance(obj, dict):
        for key in preferred:
            if key in obj:
                arr = _to_numpy(obj[key])
                if arr.ndim in (3, 4):
                    return arr
        for val in obj.values():
            found = _find_first_image(val)
            if found is not None:
                return found
    else:
        arr = _to_numpy(obj)
        if arr.ndim in (3, 4):
            return arr
    return None


def _extract_action(step: Dict[str, Any], action_dim: int) -> np.ndarray:
    action = step.get("action", None)
    if action is None:
        return np.zeros((action_dim,), dtype=np.float32)
    arr = _to_numpy(action).astype(np.float32).reshape(-1)
    if arr.shape[0] >= action_dim:
        return arr[:action_dim]
    out = np.zeros((action_dim,), dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def _normalize_chw(frame: np.ndarray, image_size: int, channels: int) -> np.ndarray:
    arr = frame[0] if frame.ndim == 4 else frame
    return _normalize_frame(arr, image_size=image_size, channels=channels)


class DROIDDataset(Dataset):
    """Episode sampler for DROID RLDS."""

    def __init__(self, episodes: List[dict], clip_len: int, image_size: int, channels: int, action_dim: int, seed: int = 0):
        self.episodes = episodes
        self.clip_len = clip_len
        self.image_size = image_size
        self.channels = channels
        self.action_dim = action_dim
        self.seed = seed

    def __len__(self) -> int:
        return max(1, len(self.episodes) * 8)

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        ep = self.episodes[int(rng.integers(0, len(self.episodes)))]
        steps = list(ep.get("steps", []))
        if len(steps) < self.clip_len:
            raise ValueError(f"DROID episode too short: {len(steps)} < clip_len={self.clip_len}")

        max_start = len(steps) - self.clip_len
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        chosen = steps[start : start + self.clip_len]

        frames = []
        actions = []
        for t, step in enumerate(chosen):
            image = _find_first_image(step.get("observation", {}))
            if image is None:
                raise ValueError("Missing observation image in DROID step.")
            frames.append(_normalize_chw(image, image_size=self.image_size, channels=self.channels))
            if t < self.clip_len - 1:
                actions.append(_extract_action(step, self.action_dim))

        return {
            "frames": torch.from_numpy(np.stack(frames, axis=0).astype(np.float32)),
            "actions": torch.from_numpy(np.stack(actions, axis=0).astype(np.float32)),
            "meta": {"dataset": "droid"},
        }


def build_droid_dataloader(
    data_root: str,
    split: str,
    batch_size: int,
    clip_len: int,
    image_size: int,
    action_dim: int,
    channels: int = 3,
    num_workers: int = 0,
    max_episodes: int | None = None,
) -> DataLoader:
    """Build DROID dataloader via TFDS RLDS episodes."""
    root = validate_required_path(data_root, "DROID", DROID_LAYOUT)
    tfds = _optional_import_tfds()

    ds = tfds.load("droid", split=split, data_dir=str(root))
    episodes = list(tfds.as_numpy(ds))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise RuntimeError(f"No DROID episodes found under {root}. Expected layout:\n{DROID_LAYOUT}")

    dataset = DROIDDataset(
        episodes=episodes,
        clip_len=clip_len,
        image_size=image_size,
        channels=channels,
        action_dim=action_dim,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_trajectory_batch,
        drop_last=True,
    )
