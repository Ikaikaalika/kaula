from __future__ import annotations

from typing import Iterator

from ..data import batch_iterator


def build_toy_iterator(cfg, num_batches: int = 100) -> Iterator[dict]:
    return batch_iterator(num_batches=num_batches, batch_size=cfg.batch_size, seq_len=cfg.seq_len, image_size=cfg.image_size, channels=cfg.channels)
