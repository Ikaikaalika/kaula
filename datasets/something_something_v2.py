"""Something-Something V2 loader for temporal reasoning supervision.

Expected layout:

    something-something-v2/
      videos/
        *.webm or *.mp4

Action labels from metadata can be integrated in later phases.
TODO(phase2): parse official labels and map to action-conditioned trajectories.
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

SSV2_LAYOUT = "something-something-v2/videos/*.webm (or *.mp4)"


def build_something_something_v2_dataloader(
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
    """Build Something-Something V2 dataloader from local videos."""
    root = validate_required_path(data_root, "Something-Something V2", SSV2_LAYOUT)
    videos = recursive_find_media(root, VIDEO_EXTS)
    if not videos:
        raise RuntimeError(
            f"No Something-Something V2 videos found under {root}. Expected layout:\n{SSV2_LAYOUT}"
        )

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
