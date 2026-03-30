"""Generate teacher distillation targets for dataset clips.

Example:
    python training/distill.py \
        --model-config configs/model_small.yaml \
        --training-config configs/training.yaml \
        --output distillation_targets.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import DatasetRequest, build_dataloader
from distillation.action_label_pipeline import generate_action_targets
from distillation.caption_pipeline import generate_caption_targets
from distillation.reward_model import score_with_teacher
from distillation.teacher_interface import TeacherInterface
from training.config_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate distillation targets")
    parser.add_argument("--model-config", default="configs/model_small.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--output", default="outputs/distillation_targets.jsonl")
    parser.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    parser.add_argument("--teacher-model", default="qwen-placeholder")
    return parser.parse_args()


def _tensor_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> None:
    args = parse_args()
    cfg_model = load_yaml(args.model_config)
    cfg_train = load_yaml(args.training_config)

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

    teacher = TeacherInterface(model_name=args.teacher_model)
    rows = []

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        frames = _tensor_to_numpy(batch["frames"])
        actions = _tensor_to_numpy(batch["actions"])

        for b in range(frames.shape[0]):
            clip_frames = [frames[b, t] for t in range(frames.shape[1])]
            clip_actions = [actions[b, t].tolist() for t in range(actions.shape[1])]

            caption_targets = generate_caption_targets(teacher=teacher, frames=clip_frames, actions=clip_actions)
            action_targets = generate_action_targets(teacher=teacher, frames=clip_frames, actions=clip_actions)
            reward_targets = score_with_teacher(teacher=teacher, trajectory=clip_frames)

            row = {
                "dataset": req.name,
                "sample_index": len(rows),
                **caption_targets,
                **action_targets,
                "trajectory_score": reward_targets,
            }
            rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "jsonl":
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    else:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(
                "Parquet export requires pandas/pyarrow. Install them or use --format jsonl."
            ) from exc
        pd.DataFrame(rows).to_parquet(out_path, index=False)

    print(f"wrote {len(rows)} distillation records to {out_path}")


if __name__ == "__main__":
    main()
