"""Dataset registry for world-model training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from torch.utils.data import DataLoader

from .bridge_data import build_bridge_data_dataloader
from .droid import build_droid_dataloader
from .ego4d import build_ego4d_dataloader
from .epic_kitchens import build_epic_kitchens_dataloader
from .something_something_v2 import build_something_something_v2_dataloader
from .toy_dataset import build_toy_dataloader


@dataclass
class DatasetRequest:
    """Canonical dataset request used by scripts."""

    name: str
    data_root: str | None
    split: str
    batch_size: int
    clip_len: int
    image_size: int
    action_dim: int
    channels: int
    stride: int
    num_workers: int
    max_episodes: int | None = None


SUPPORTED_DATASETS: Dict[str, str] = {
    "toy": "Synthetic moving-shapes data for smoke testing.",
    "epic_kitchens": "EPIC-KITCHENS frame folders.",
    "ego4d": "Ego4D video clips.",
    "droid": "DROID RLDS episodes with action trajectories.",
    "bridge_data": "BridgeData V2 trajectories.",
    "something_something_v2": "Something-Something V2 videos.",
}


def build_dataloader(req: DatasetRequest) -> DataLoader:
    """Construct a dataset dataloader from canonical request."""
    name = req.name.lower().strip()
    if name == "toy":
        return build_toy_dataloader(
            batch_size=req.batch_size,
            clip_len=req.clip_len,
            image_size=req.image_size,
            action_dim=req.action_dim,
            channels=req.channels,
            num_workers=req.num_workers,
        )
    if name == "epic_kitchens":
        return build_epic_kitchens_dataloader(
            data_root=req.data_root or "",
            batch_size=req.batch_size,
            clip_len=req.clip_len,
            image_size=req.image_size,
            action_dim=req.action_dim,
            channels=req.channels,
            stride=req.stride,
            num_workers=req.num_workers,
        )
    if name == "ego4d":
        return build_ego4d_dataloader(
            data_root=req.data_root or "",
            batch_size=req.batch_size,
            clip_len=req.clip_len,
            image_size=req.image_size,
            action_dim=req.action_dim,
            channels=req.channels,
            stride=req.stride,
            num_workers=req.num_workers,
        )
    if name == "droid":
        return build_droid_dataloader(
            data_root=req.data_root or "",
            split=req.split,
            batch_size=req.batch_size,
            clip_len=req.clip_len,
            image_size=req.image_size,
            action_dim=req.action_dim,
            channels=req.channels,
            num_workers=req.num_workers,
            max_episodes=req.max_episodes,
        )
    if name == "bridge_data":
        return build_bridge_data_dataloader(
            data_root=req.data_root or "",
            batch_size=req.batch_size,
            clip_len=req.clip_len,
            image_size=req.image_size,
            action_dim=req.action_dim,
            channels=req.channels,
            stride=req.stride,
            num_workers=req.num_workers,
        )
    if name == "something_something_v2":
        return build_something_something_v2_dataloader(
            data_root=req.data_root or "",
            batch_size=req.batch_size,
            clip_len=req.clip_len,
            image_size=req.image_size,
            action_dim=req.action_dim,
            channels=req.channels,
            stride=req.stride,
            num_workers=req.num_workers,
        )

    raise ValueError(f"Unsupported dataset '{req.name}'. Supported: {sorted(SUPPORTED_DATASETS)}")
