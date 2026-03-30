"""Shared dataset helpers for clip sampling and validation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _normalize_frame(frame: np.ndarray, image_size: int, channels: int) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected frame with rank 3 [H,W,C], got {arr.shape}")

    if arr.dtype != np.uint8:
        max_val = float(arr.max()) if arr.size else 1.0
        if max_val <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        pil = Image.fromarray(arr[..., 0], mode="L")
    else:
        pil = Image.fromarray(arr[..., :3], mode="RGB")

    pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(pil)

    if arr.ndim == 2:
        arr = arr[..., None]

    if channels == 1 and arr.shape[-1] > 1:
        arr = arr[..., :3].mean(axis=-1, keepdims=True)
    elif channels == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, repeats=3, axis=-1)
    elif arr.shape[-1] > channels:
        arr = arr[..., :channels]

    arr = arr.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _read_video_random_clip(
    video_path: Path,
    clip_len: int,
    image_size: int,
    channels: int,
    stride: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frames = list(iio.imiter(str(video_path)))
    required = clip_len * stride
    if len(frames) < required:
        raise ValueError(
            f"Video '{video_path}' has {len(frames)} frames; requires at least {required} for clip_len={clip_len}, stride={stride}."
        )
    max_start = len(frames) - required
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    chosen = [frames[start + i * stride] for i in range(clip_len)]
    return np.stack([_normalize_frame(f, image_size=image_size, channels=channels) for f in chosen], axis=0)


def _read_frame_dir_random_clip(
    frame_dir: Path,
    clip_len: int,
    image_size: int,
    channels: int,
    stride: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frame_paths = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
    required = clip_len * stride
    if len(frame_paths) < required:
        raise ValueError(
            f"Frame folder '{frame_dir}' has {len(frame_paths)} frames; requires at least {required} for clip_len={clip_len}, stride={stride}."
        )
    max_start = len(frame_paths) - required
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    chosen = [frame_paths[start + i * stride] for i in range(clip_len)]
    clip = [iio.imread(str(path)) for path in chosen]
    return np.stack([_normalize_frame(f, image_size=image_size, channels=channels) for f in clip], axis=0)


def recursive_find_media(root: Path, exts: set[str]) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


class VideoClipDataset(Dataset):
    """Random clip sampler from a list of video files."""

    def __init__(
        self,
        video_paths: List[Path],
        clip_len: int,
        image_size: int,
        channels: int,
        action_dim: int,
        stride: int = 1,
        dataset_size: int = 10_000,
        seed: int = 0,
    ) -> None:
        self.video_paths = video_paths
        self.clip_len = clip_len
        self.image_size = image_size
        self.channels = channels
        self.action_dim = action_dim
        self.stride = stride
        self.dataset_size = dataset_size
        self.seed = seed

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        video_path = self.video_paths[int(rng.integers(0, len(self.video_paths)))]
        clip = _read_video_random_clip(
            video_path=video_path,
            clip_len=self.clip_len,
            image_size=self.image_size,
            channels=self.channels,
            stride=self.stride,
            rng=rng,
        )
        actions = np.zeros((self.clip_len - 1, self.action_dim), dtype=np.float32)
        return {
            "frames": torch.from_numpy(clip),
            "actions": torch.from_numpy(actions),
            "meta": {"source": str(video_path)},
        }


class FrameFolderClipDataset(Dataset):
    """Random clip sampler from frame directories."""

    def __init__(
        self,
        frame_dirs: List[Path],
        clip_len: int,
        image_size: int,
        channels: int,
        action_dim: int,
        stride: int = 1,
        dataset_size: int = 10_000,
        seed: int = 0,
    ) -> None:
        self.frame_dirs = frame_dirs
        self.clip_len = clip_len
        self.image_size = image_size
        self.channels = channels
        self.action_dim = action_dim
        self.stride = stride
        self.dataset_size = dataset_size
        self.seed = seed

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        frame_dir = self.frame_dirs[int(rng.integers(0, len(self.frame_dirs)))]
        clip = _read_frame_dir_random_clip(
            frame_dir=frame_dir,
            clip_len=self.clip_len,
            image_size=self.image_size,
            channels=self.channels,
            stride=self.stride,
            rng=rng,
        )
        actions = np.zeros((self.clip_len - 1, self.action_dim), dtype=np.float32)
        return {
            "frames": torch.from_numpy(clip),
            "actions": torch.from_numpy(actions),
            "meta": {"source": str(frame_dir)},
        }


def collate_trajectory_batch(items: list[dict]) -> dict:
    frames = torch.stack([x["frames"] for x in items], dim=0)
    actions = torch.stack([x["actions"] for x in items], dim=0)
    meta = [x.get("meta", {}) for x in items]
    return {"frames": frames, "actions": actions, "meta": meta}


def validate_required_path(path: Optional[str], dataset_name: str, expected_layout: str) -> Path:
    if not path:
        raise ValueError(
            f"{dataset_name} requires a data root path. Expected layout:\n{expected_layout}"
        )
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"{dataset_name} data root does not exist: {root}\nExpected layout:\n{expected_layout}"
        )
    return root
