from __future__ import annotations

from .toy import build_toy_iterator
from .epic_kitchens import build_epic_kitchens_iterator
from .ego4d import build_ego4d_iterator
from .droid import build_droid_iterator


def describe_supported_datasets():
    return {
        "toy": "Built-in moving-shapes synthetic dataset; no external files required.",
        "epic_kitchens": "Reads local EPIC-KITCHENS RGB frame folders.",
        "ego4d": "Reads local Ego4D mp4 clips/videos downloaded via the official CLI.",
        "droid": "Reads local DROID RLDS episodes via TensorFlow Datasets (optional dependency).",
    }


def build_dataset_iterator(cfg, num_batches: int = 100):
    name = (cfg.dataset_name or "toy").lower()
    if name == "toy":
        return build_toy_iterator(cfg, num_batches=num_batches)
    if name == "epic_kitchens":
        return build_epic_kitchens_iterator(cfg, num_batches=num_batches)
    if name == "ego4d":
        return build_ego4d_iterator(cfg, num_batches=num_batches)
    if name == "droid":
        return build_droid_iterator(cfg, num_batches=num_batches)
    raise ValueError(f"Unsupported dataset_name={cfg.dataset_name}")
