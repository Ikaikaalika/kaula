"""Train latent state-space diffusion world model.

Example:
    python training/train.py \\
        --model-config configs/model_small.yaml \\
        --diffusion-config configs/diffusion.yaml \\
        --training-config configs/training.yaml \\
        --max-steps 20
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
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


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    factor = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return base_lr * factor


def teacher_forcing_prob(step: int, decay_steps: int, min_prob: float) -> float:
    """Linear decay from 1.0 to `min_prob` over `decay_steps`."""
    if decay_steps <= 0:
        return min_prob
    progress = min(step / decay_steps, 1.0)
    return 1.0 - progress * (1.0 - min_prob)


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.model_config, args.diffusion_config, args.training_config)
    cfg_model, cfg_diff, cfg_train = cfg.model, cfg.diffusion, cfg.training

    max_steps = int(args.max_steps or cfg_train.get("max_steps", 100))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    seed = int(cfg_train.get("seed", 7))
    set_seed(seed)

    model = build_model(cfg_model, cfg_diff).to(device)
    model.train()

    ema_decay = float(cfg_train.get("ema_decay", 0.0))
    ema = EMA(model, ema_decay) if ema_decay > 0.0 else None

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
        latent_prior=float(cfg_train.get("lambda_latent_prior", 0.5)),
        distillation=float(cfg_train.get("lambda_distillation", 0.0)),
        contrastive_alignment=float(cfg_train.get("lambda_contrastive_alignment", 0.0)),
    )

    base_lr = float(cfg_train.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(cfg_train.get("weight_decay", 1e-2)),
    )
    warmup_steps = int(cfg_train.get("lr_warmup_steps", min(100, max_steps // 10)))
    min_lr_ratio = float(cfg_train.get("lr_min_ratio", 0.1))

    tf_decay_steps = int(cfg_train.get("teacher_forcing_decay_steps", max(1, max_steps // 2)))
    tf_min_prob = float(cfg_train.get("teacher_forcing_min", 0.5))

    use_amp = bool(cfg_train.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_dtype = torch.float16

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

        lr_now = cosine_lr(step, base_lr, warmup_steps, max_steps, min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        tf_prob = teacher_forcing_prob(step, tf_decay_steps, tf_min_prob)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model.forward_train(frames=frames, actions=actions, teacher_forcing_prob=tf_prob)
                loss_dict = compute_world_model_losses(outputs=outputs, weights=weights)
                loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg_train.get("grad_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model.forward_train(frames=frames, actions=actions, teacher_forcing_prob=tf_prob)
            loss_dict = compute_world_model_losses(outputs=outputs, weights=weights)
            loss = loss_dict["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg_train.get("grad_clip_norm", 1.0)))
            optimizer.step()

        if ema is not None:
            ema.update(model)

        row = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}
        row["step"] = step
        row["lr"] = lr_now
        row["teacher_forcing_prob"] = tf_prob
        logs.append(row)

        if step % log_interval == 0 or step == 1:
            print(
                f"step={step:04d} "
                f"loss={row['loss']:.5f} "
                f"diff={row['diffusion_loss']:.5f} "
                f"latent={row['latent_rollout_loss']:.5f} "
                f"recon={row['reconstruction_loss']:.5f} "
                f"prior={row['latent_prior_loss']:.5f} "
                f"lr={lr_now:.2e} tf={tf_prob:.2f}"
            )

        if step % save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_step_{step:06d}.pt"
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": cfg_model,
                "diffusion_config": cfg_diff,
                "training_config": cfg_train,
            }
            if ema is not None:
                ckpt["ema_state_dict"] = ema.shadow.state_dict()
            torch.save(ckpt, ckpt_path)

    final_ckpt = output_dir / "model_final.pt"
    final = {
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": cfg_model,
        "diffusion_config": cfg_diff,
        "training_config": cfg_train,
    }
    if ema is not None:
        final["ema_state_dict"] = ema.shadow.state_dict()
    torch.save(final, final_ckpt)

    log_path = output_dir / "train_logs.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for row in logs:
            fh.write(json.dumps(row) + "\n")

    print(f"saved final checkpoint: {final_ckpt}")
    print(f"saved logs: {log_path}")


if __name__ == "__main__":
    main()
