"""Ego4D dataset loader.

Expected layout (example):

    ego4d/
      v1/
        clips/
          *.mp4
      videos/
        *.mp4

The loader samples clips of length T from available video files.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from .common import (
    VIDEO_EXTS,
    VideoClipDataset,
    collate_trajectory_batch,
    recursive_find_media,
    validate_required_path,
)

EGO4D_LAYOUT = "ego4d/<split>/clips/*.mp4 or ego4d/videos/*.mp4"


def build_ego4d_dataloader(
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
    """Build Ego4D dataloader from local video files."""
    root = validate_required_path(data_root, "Ego4D", EGO4D_LAYOUT)
    videos = recursive_find_media(root, VIDEO_EXTS)
    if not videos:
        raise RuntimeError(f"No Ego4D video files found under {root}. Expected layout:\n{EGO4D_LAYOUT}")

    ds = VideoClipDataset(
        video_paths=videos,
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
