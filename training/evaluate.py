"""Evaluate latent state-space diffusion world model.

Example:
    python training/evaluate.py \
        --checkpoint outputs/phase1_train/model_final.pt \
        --model-config configs/model_small.yaml \
        --diffusion-config configs/diffusion.yaml \
        --training-config configs/training.yaml
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent state-space diffusion world model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="configs/model_small.yaml")
    parser.add_argument("--diffusion-config", default="configs/diffusion.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--output", default="outputs/eval_metrics.json")
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


def psnr_from_mse(mse_value: float, max_value: float = 1.0) -> float:
    if mse_value <= 0:
        return float("inf")
    return 20.0 * np.log10(max_value) - 10.0 * np.log10(mse_value)


def ssim_placeholder() -> float:
    """Placeholder SSIM implementation.

    TODO(phase5): replace with multi-scale SSIM from torchmetrics or skimage.
    """
    return float("nan")


def lpips_placeholder() -> float:
    """Placeholder LPIPS implementation.

    TODO(phase5): integrate LPIPS network.
    """
    return float("nan")


def fvd_placeholder() -> float:
    """Placeholder FVD implementation.

    TODO(phase5): integrate I3D-based Frechet Video Distance.
    """
    return float("nan")


def trajectory_consistency(pred_latents: torch.Tensor, target_latents: torch.Tensor, eps: float = 1e-8) -> float:
    pred_delta = pred_latents[:, 1:] - pred_latents[:, :-1]
    tgt_delta = target_latents[:, 1:] - target_latents[:, :-1]
    pred_norm = pred_delta / (pred_delta.norm(dim=-1, keepdim=True) + eps)
    tgt_norm = tgt_delta / (tgt_delta.norm(dim=-1, keepdim=True) + eps)
    score = (pred_norm * tgt_norm).sum(dim=-1).mean()
    return float(score.detach().cpu().item())


def latent_cosine(pred_latents: torch.Tensor, target_latents: torch.Tensor, eps: float = 1e-8) -> float:
    pred_norm = pred_latents / (pred_latents.norm(dim=-1, keepdim=True) + eps)
    tgt_norm = target_latents / (target_latents.norm(dim=-1, keepdim=True) + eps)
    score = (pred_norm * tgt_norm).sum(dim=-1).mean()
    return float(score.detach().cpu().item())


def main() -> None:
    args = parse_args()
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
        batch_size=int(cfg_train.get("batch_size", 8)),
        clip_len=int(cfg_model.get("clip_len", 8)),
        image_size=int(cfg_model.get("image_size", 64)),
        action_dim=int(cfg_model.get("action_dim", 4)),
        channels=int(cfg_model.get("channels", 3)),
        stride=int(cfg_train.get("sample_stride", 1)),
        num_workers=int(cfg_train.get("num_workers", 0)),
        max_episodes=cfg_train.get("max_episodes", None),
    )
    loader = build_dataloader(req)

    mse_recon_values = []
    cosine_values = []
    consistency_values = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            out = model.forward_train(frames=frames, actions=actions)

            mse_recon = torch.mean((out["decoded_frames"] - out["target_frames"]) ** 2)
            mse_recon_values.append(float(mse_recon.detach().cpu().item()))
            cosine_values.append(latent_cosine(out["pred_latents"], out["target_latents"]))
            consistency_values.append(trajectory_consistency(out["pred_latents"], out["target_latents"]))

    recon_mse = float(np.mean(mse_recon_values)) if mse_recon_values else float("nan")
    metrics = {
        "psnr": psnr_from_mse(recon_mse),
        "ssim": ssim_placeholder(),
        "lpips": lpips_placeholder(),
        "fvd": fvd_placeholder(),
        "latent_cosine_similarity": float(np.mean(cosine_values)) if cosine_values else float("nan"),
        "trajectory_consistency_score": float(np.mean(consistency_values)) if consistency_values else float("nan"),
        "semantic_alignment_score": float("nan"),  # TODO(phase5): teacher-based optional alignment.
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"saved metrics: {out_path}")


if __name__ == "__main__":
    main()
