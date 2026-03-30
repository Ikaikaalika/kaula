from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Optional, List

import imageio.v3 as iio
import numpy as np
import mlx.core as mx

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def normalize_frame(frame: np.ndarray, channels: int = 1, image_size: int = 32) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC frame, got shape {arr.shape}")

    if arr.shape[-1] >= 3 and channels == 1:
        arr = arr[..., :3].mean(axis=-1, keepdims=True)
    elif arr.shape[-1] == 1 and channels == 3:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > channels:
        arr = arr[..., :channels]

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    h, w = arr.shape[:2]
    if h != image_size or w != image_size:
        y_idx = np.linspace(0, h - 1, image_size).astype(np.int32)
        x_idx = np.linspace(0, w - 1, image_size).astype(np.int32)
        arr = arr[y_idx][:, x_idx]
    return arr


def read_video_clip(video_path: str | Path, seq_len: int, image_size: int, channels: int, stride: int = 1, start_frame: Optional[int] = None) -> np.ndarray:
    video_path = str(video_path)
    props = iio.improps(video_path)
    nframes = props.n_images
    if nframes is None or nframes <= 0:
        frames = list(iio.imiter(video_path))
        nframes = len(frames)
        use_cache = True
    else:
        use_cache = False
    if nframes < seq_len * stride:
        raise ValueError(f"Video {video_path} too short for seq_len={seq_len}, stride={stride}")
    if start_frame is None:
        max_start = max(0, nframes - seq_len * stride)
        start_frame = 0 if max_start == 0 else np.random.randint(0, max_start + 1)
    frame_ids = [start_frame + i * stride for i in range(seq_len)]
    clip = []
    if use_cache:
        for idx in frame_ids:
            clip.append(normalize_frame(frames[idx], channels=channels, image_size=image_size))
    else:
        for idx in frame_ids:
            clip.append(normalize_frame(iio.imread(video_path, index=idx), channels=channels, image_size=image_size))
    return np.stack(clip, axis=0)


def read_frame_dir_clip(frame_dir: str | Path, seq_len: int, image_size: int, channels: int, stride: int = 1, start_idx: Optional[int] = None) -> np.ndarray:
    frame_dir = Path(frame_dir)
    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if len(frames) < seq_len * stride:
        raise ValueError(f"Frame dir {frame_dir} too short for seq_len={seq_len}, stride={stride}")
    if start_idx is None:
        max_start = max(0, len(frames) - seq_len * stride)
        start_idx = 0 if max_start == 0 else np.random.randint(0, max_start + 1)
    selected = [frames[start_idx + i * stride] for i in range(seq_len)]
    clip = [normalize_frame(iio.imread(str(p)), channels=channels, image_size=image_size) for p in selected]
    return np.stack(clip, axis=0)


def zero_actions(batch_size: int, seq_len: int, action_dim: int) -> mx.array:
    return mx.zeros((batch_size, seq_len - 1, action_dim), dtype=mx.float32)


def batch_dict(frames: np.ndarray, action_dim: int) -> Dict[str, mx.array]:
    b, t = frames.shape[:2]
    return {"frames": mx.array(frames.astype(np.float32)), "actions": zero_actions(b, t, action_dim)}


def recursive_find_media(root: Path, exts: set[str]) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
