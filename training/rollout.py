"""Sample world-model rollouts from a trained checkpoint.

Example:
    python training/rollout.py \
        --checkpoint outputs/phase1_train/model_final.pt \
        --horizon 32 \
        --num-samples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import DatasetRequest, build_dataloader
from training.config_utils import load_experiment_config
from training.model_factory import build_model


VALID_HORIZONS = {1, 8, 32, 128}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout sampler for latent state-space diffusion model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="configs/model_small.yaml")
    parser.add_argument("--diffusion-config", default="configs/diffusion.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--output", default="outputs/rollout_sample.npz")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def choose_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _expand_actions(actions: torch.Tensor, horizon: int) -> torch.Tensor:
    if actions.shape[1] >= horizon:
        return actions[:, :horizon]
    last = actions[:, -1:]
    repeat = horizon - actions.shape[1]
    tail = last.repeat(1, repeat, 1)
    return torch.cat([actions, tail], dim=1)


def main() -> None:
    args = parse_args()
    if args.horizon not in VALID_HORIZONS:
        raise ValueError(f"Unsupported horizon={args.horizon}. Expected one of {sorted(VALID_HORIZONS)}")

    cfg = load_experiment_config(args.model_config, args.diffusion_config, args.training_config)
    cfg_model, cfg_diff, cfg_train = cfg.model, cfg.diffusion, cfg.training

    device = choose_device(args.device)
    model = build_model(cfg_model, cfg_diff).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    req = DatasetRequest(
        name=str(cfg_train.get("dataset_name", "toy")),
        data_root=cfg_train.get("data_root", None),
        split=str(cfg_train.get("split", "train")),
        batch_size=1,
        clip_len=int(cfg_model.get("clip_len", 8)),
        image_size=int(cfg_model.get("image_size", 64)),
        action_dim=int(cfg_model.get("action_dim", 4)),
        channels=int(cfg_model.get("channels", 3)),
        stride=int(cfg_train.get("sample_stride", 1)),
        num_workers=0,
        max_episodes=cfg_train.get("max_episodes", None),
    )
    loader = build_dataloader(req)
    batch = next(iter(loader))

    frames = batch["frames"].to(device)
    actions = batch["actions"].to(device)

    context_len = min(args.context_len, frames.shape[1])
    context = frames[:, :context_len]
    rollout_actions = _expand_actions(actions, horizon=args.horizon)

    with torch.no_grad():
        out = model.rollout(
            context_frames=context,
            action_sequence=rollout_actions,
            rollout_horizon=args.horizon,
            num_samples=args.num_samples,
        )

    latent_traj = out["latent_trajectory"].detach().cpu().numpy()
    decoded_traj = out["decoded_trajectory"].detach().cpu().numpy()
    uncertainty = out["uncertainty"].detach().cpu().numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        latent_trajectory=latent_traj,
        decoded_trajectory=decoded_traj,
        uncertainty=uncertainty,
    )
    print(f"saved rollout sample: {output_path}")
    print(f"latent trajectory shape: {latent_traj.shape}")
    print(f"decoded trajectory shape: {decoded_traj.shape}")
    print(f"uncertainty shape: {uncertainty.shape}")


if __name__ == "__main__":
    main()
