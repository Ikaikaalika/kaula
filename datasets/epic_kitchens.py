"""EPIC-KITCHENS dataset loader.

Expected layout (example):

    EPIC-KITCHENS/
      P01/
        rgb_frames/
          P01_01/
            frame_0000000001.jpg
            ...

The loader samples clips of length T from frame directories.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .common import (
    FrameFolderClipDataset,
    collate_trajectory_batch,
    recursive_find_media,
    validate_required_path,
    IMAGE_EXTS,
)

EPIC_LAYOUT = "EPIC-KITCHENS/<participant>/rgb_frames/<video_id>/frame_*.jpg"


def _find_frame_dirs(root: Path) -> list[Path]:
    frame_files = recursive_find_media(root, IMAGE_EXTS)
    return sorted({f.parent for f in frame_files if f.name.lower().startswith("frame_") or f.suffix.lower() in IMAGE_EXTS})


def build_epic_kitchens_dataloader(
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
    """Build EPIC-KITCHENS dataloader from local frame folders."""
    root = validate_required_path(data_root, "EPIC-KITCHENS", EPIC_LAYOUT)
    frame_dirs = _find_frame_dirs(root)
    if not frame_dirs:
        raise RuntimeError(
            f"No EPIC-KITCHENS frame folders found under {root}. Expected layout:\n{EPIC_LAYOUT}"
        )

    ds = FrameFolderClipDataset(
        frame_dirs=frame_dirs,
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
