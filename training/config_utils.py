"""Configuration loading helpers for YAML-based experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ExperimentConfig:
    """Merged experiment configuration."""

    model: Dict[str, Any]
    diffusion: Dict[str, Any]
    training: Dict[str, Any]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {p}, got {type(data)}")
    return data


def load_experiment_config(model_path: str, diffusion_path: str, training_path: str) -> ExperimentConfig:
    """Load and merge model/diffusion/training configs."""
    model_cfg = load_yaml(model_path)
    diffusion_cfg = load_yaml(diffusion_path)
    training_cfg = load_yaml(training_path)
    return ExperimentConfig(model=model_cfg, diffusion=diffusion_cfg, training=training_cfg)
