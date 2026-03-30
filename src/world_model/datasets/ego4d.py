from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .common import batch_dict, read_video_clip, recursive_find_media, VIDEO_EXTS


def build_ego4d_iterator(cfg, num_batches: int = 100) -> Iterator[dict]:
    if not cfg.data_root:
        raise ValueError("Ego4D loader requires cfg.data_root")
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(root)
    videos = recursive_find_media(root, VIDEO_EXTS)
    if cfg.max_videos:
        videos = videos[: cfg.max_videos]
    if not videos:
        raise RuntimeError(f"No Ego4D video files found under {root}")
    for _ in range(num_batches):
        batch_frames = []
        while len(batch_frames) < cfg.batch_size:
            video_path = videos[np.random.randint(0, len(videos))]
            try:
                clip = read_video_clip(video_path, seq_len=cfg.seq_len, image_size=cfg.image_size, channels=cfg.channels, stride=cfg.sample_stride)
                batch_frames.append(clip)
            except Exception:
                continue
        yield batch_dict(np.stack(batch_frames, axis=0), action_dim=cfg.action_dim)
