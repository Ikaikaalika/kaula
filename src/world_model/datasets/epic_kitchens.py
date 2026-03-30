from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np

from .common import batch_dict, read_frame_dir_clip, IMAGE_EXTS


def _find_epic_clip_dirs(root: Path) -> List[Path]:
    candidates = []
    for p in root.rglob("*"):
        if p.is_dir():
            try:
                if any(x.name.startswith("frame_") for x in p.iterdir() if x.is_file()):
                    candidates.append(p)
            except PermissionError:
                continue
    if not candidates:
        for p in root.rglob("*"):
            if p.is_dir():
                try:
                    if any(x.suffix.lower() in IMAGE_EXTS for x in p.iterdir() if x.is_file()):
                        candidates.append(p)
                except PermissionError:
                    continue
    return sorted(set(candidates))


def build_epic_kitchens_iterator(cfg, num_batches: int = 100) -> Iterator[dict]:
    if not cfg.data_root:
        raise ValueError("EPIC-KITCHENS loader requires cfg.data_root")
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(root)
    clip_dirs = _find_epic_clip_dirs(root)
    if cfg.max_videos:
        clip_dirs = clip_dirs[: cfg.max_videos]
    if not clip_dirs:
        raise RuntimeError(f"No EPIC-KITCHENS frame directories found under {root}")
    for _ in range(num_batches):
        batch_frames = []
        while len(batch_frames) < cfg.batch_size:
            clip_dir = clip_dirs[np.random.randint(0, len(clip_dirs))]
            try:
                clip = read_frame_dir_clip(clip_dir, seq_len=cfg.seq_len, image_size=cfg.image_size, channels=cfg.channels, stride=cfg.sample_stride)
                batch_frames.append(clip)
            except Exception:
                continue
        yield batch_dict(np.stack(batch_frames, axis=0), action_dim=cfg.action_dim)
