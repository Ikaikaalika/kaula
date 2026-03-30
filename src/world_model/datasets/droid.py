from __future__ import annotations

from typing import Any, Dict, Iterator

import numpy as np

from .common import normalize_frame


def _optional_import_tfds():
    try:
        import tensorflow_datasets as tfds
        return tfds
    except Exception as exc:
        raise ImportError("DROID RLDS loader requires tensorflow_datasets and tensorflow. Install optional deps from requirements-datasets.txt") from exc


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _find_first_image(obj: Any):
    preferred_keys = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left", "wrist_image_right", "image", "rgb", "pixels"]
    if isinstance(obj, dict):
        for k in preferred_keys:
            if k in obj:
                arr = _to_numpy(obj[k])
                if arr.ndim in (3, 4):
                    return arr
        for v in obj.values():
            found = _find_first_image(v)
            if found is not None:
                return found
    else:
        arr = _to_numpy(obj)
        if arr.ndim in (3, 4):
            return arr
    return None


def _extract_action(step: Dict[str, Any], action_dim: int):
    a = step.get("action", None)
    if a is None:
        return np.zeros((action_dim,), dtype=np.float32)
    a = _to_numpy(a).astype(np.float32).reshape(-1)
    if a.shape[0] >= action_dim:
        return a[:action_dim]
    out = np.zeros((action_dim,), dtype=np.float32)
    out[: a.shape[0]] = a
    return out


def build_droid_iterator(cfg, num_batches: int = 100) -> Iterator[dict]:
    tfds = _optional_import_tfds()
    ds = tfds.load("droid", split=cfg.split or "train", data_dir=cfg.data_root)
    episodes = list(tfds.as_numpy(ds))
    if cfg.max_videos:
        episodes = episodes[: cfg.max_videos]
    if not episodes:
        raise RuntimeError("No DROID episodes found")
    for _ in range(num_batches):
        batch_frames, batch_actions = [], []
        while len(batch_frames) < cfg.batch_size:
            ep = episodes[np.random.randint(0, len(episodes))]
            steps = list(ep.get("steps", []))
            if len(steps) < cfg.seq_len:
                continue
            max_start = len(steps) - cfg.seq_len
            start = 0 if max_start == 0 else np.random.randint(0, max_start + 1)
            chosen = steps[start : start + cfg.seq_len]
            frames, actions = [], []
            ok = True
            for i, step in enumerate(chosen):
                image = _find_first_image(step.get("observation", {}))
                if image is None:
                    ok = False
                    break
                if image.ndim == 4:
                    image = image[0]
                frames.append(normalize_frame(image, channels=cfg.channels, image_size=cfg.image_size))
                if i < cfg.seq_len - 1:
                    actions.append(_extract_action(step, cfg.action_dim))
            if not ok:
                continue
            batch_frames.append(np.stack(frames, axis=0))
            batch_actions.append(np.stack(actions, axis=0))
        import mlx.core as mx
        yield {"frames": mx.array(np.stack(batch_frames, axis=0).astype(np.float32)), "actions": mx.array(np.stack(batch_actions, axis=0).astype(np.float32))}
