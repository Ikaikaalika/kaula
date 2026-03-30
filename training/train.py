"""Train latent state-space diffusion world model.

Example:
    python training/train.py \
        --model-config configs/model_small.yaml \
        --diffusion-config configs/diffusion.yaml \
        --training-config configs/training.yaml \
        --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import DatasetRequest, build_dataloader
from world_model.losses import LossWeights, compute_world_model_losses

from training.config_utils import load_experiment_config
from training.model_factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent state-space diffusion world model")
    parser.add_argument("--model-config", default="configs/model_small.yaml")
    parser.add_argument("--diffusion-config", default="configs/diffusion.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--output-dir", default="outputs/phase1_train")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto if omitted)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_mlx_prototype(cfg_training: dict) -> int:
    """Run local MLX prototype mode.

    This delegates to legacy MLX path while Phase 1 focuses on PyTorch architecture wiring.
    TODO(phase3): migrate MLX mode to this same world_model package.
    """
    command = [
        sys.executable,
        str(ROOT / "run_demo.py"),
        "--dataset",
        "toy",
        "--num-batches",
        str(cfg_training.get("max_steps", 10)),
    ]
    print("[mlx] delegating to:", " ".join(command))
    result = subprocess.run(command, check=False)
    return result.returncode


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.model_config, args.diffusion_config, args.training_config)
    cfg_model, cfg_diff, cfg_train = cfg.model, cfg.diffusion, cfg.training

    max_steps = int(args.max_steps or cfg_train.get("max_steps", 100))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = str(cfg_train.get("runtime", "pytorch")).lower().strip()
    if runtime == "mlx":
        exit_code = run_mlx_prototype(cfg_train)
        if exit_code != 0:
            raise RuntimeError(f"MLX prototype run failed with exit code {exit_code}")
        return

    device = choose_device(args.device)
    seed = int(cfg_train.get("seed", 7))
    set_seed(seed)

    model = build_model(cfg_model, cfg_diff).to(device)
    model.train()

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

    weights = LossWeights(
        diffusion=float(cfg_train.get("lambda_diffusion", 1.0)),
        latent_rollout=float(cfg_train.get("lambda_latent_rollout", 1.0)),
        reconstruction=float(cfg_train.get("lambda_reconstruction", 1.0)),
        distillation=float(cfg_train.get("lambda_distillation", 0.0)),
        contrastive_alignment=float(cfg_train.get("lambda_contrastive_alignment", 0.0)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_train.get("learning_rate", 1e-4)),
        weight_decay=float(cfg_train.get("weight_decay", 1e-2)),
    )

    logs = []
    log_interval = int(cfg_train.get("log_interval", 10))
    save_interval = int(cfg_train.get("save_interval", 50))

    iterator = iter(loader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_train(frames=frames, actions=actions)
        loss_dict = compute_world_model_losses(outputs=outputs, weights=weights)
        loss = loss_dict["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg_train.get("grad_clip_norm", 1.0)))
        optimizer.step()

        row = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}
        row["step"] = step
        logs.append(row)

        if step % log_interval == 0 or step == 1:
            print(
                f"step={step:04d} "
                f"loss={row['loss']:.5f} "
                f"diff={row['diffusion_loss']:.5f} "
                f"latent={row['latent_rollout_loss']:.5f} "
                f"recon={row['reconstruction_loss']:.5f}"
            )

        if step % save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": cfg_model,
                    "diffusion_config": cfg_diff,
                    "training_config": cfg_train,
                },
                ckpt_path,
            )

    final_ckpt = output_dir / "model_final.pt"
    torch.save(
        {
            "step": max_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": cfg_model,
            "diffusion_config": cfg_diff,
            "training_config": cfg_train,
        },
        final_ckpt,
    )

    log_path = output_dir / "train_logs.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for row in logs:
            fh.write(json.dumps(row) + "\n")

    print(f"saved final checkpoint: {final_ckpt}")
    print(f"saved logs: {log_path}")


if __name__ == "__main__":
    main()
