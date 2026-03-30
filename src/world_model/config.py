from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, Optional

@dataclass
class WorldModelConfig:
    image_size: int = 32
    channels: int = 1
    seq_len: int = 8
    patch_size: int = 4

    latent_dim: int = 128
    action_dim: int = 4
    hidden_dim: int = 256
    ssm_hidden_dim: int = 128
    num_ssm_layers: int = 2
    diffusion_hidden_dim: int = 256

    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 5

    diffusion_steps: int = 32
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    recon_weight: float = 1.0
    latent_weight: float = 1.0
    diffusion_weight: float = 1.0

    seed: int = 42

    dataset_name: str = "toy"
    data_root: Optional[str] = None
    annotation_path: Optional[str] = None
    split: str = "train"
    sample_stride: int = 1
    max_videos: Optional[int] = None


def _validate_keys(values: dict[str, Any]) -> None:
    allowed = {f.name for f in fields(WorldModelConfig)}
    unknown = sorted(set(values.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")


def config_from_dict(values: dict[str, Any]) -> WorldModelConfig:
    _validate_keys(values)
    cleaned = dict(values)
    if isinstance(cleaned.get("data_root"), str):
        cleaned["data_root"] = str(Path(cleaned["data_root"]).expanduser())
    if isinstance(cleaned.get("annotation_path"), str):
        cleaned["annotation_path"] = str(Path(cleaned["annotation_path"]).expanduser())
    return WorldModelConfig(**cleaned)


def load_config_json(path: str | Path) -> WorldModelConfig:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return config_from_dict(payload)


def apply_overrides(cfg: WorldModelConfig, **overrides: Any) -> WorldModelConfig:
    allowed = {f.name for f in fields(WorldModelConfig)}
    updates = {}
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f"Unsupported override key: {key}")
        if value is not None:
            updates[key] = value
    if not updates:
        return cfg
    return replace(cfg, **updates)
