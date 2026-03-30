from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from world_model.config import WorldModelConfig, apply_overrides, load_config_json
from world_model.data import set_seed
from world_model.train import make_model_and_optimizer
from world_model.evaluate import evaluate_one_step

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the minimal MLX world model")
    p.add_argument("--config", default=None, help="Path to JSON config file (see configs/)")
    p.add_argument("--dataset", default=None, choices=["toy", "epic_kitchens", "ego4d", "droid"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--annotation-path", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--num-batches", type=int, default=10)
    p.add_argument("--sample-stride", type=int, default=None)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--split", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    base_cfg = load_config_json(args.config) if args.config else WorldModelConfig()
    cfg = apply_overrides(
        base_cfg,
        dataset_name=args.dataset,
        data_root=args.data_root,
        annotation_path=args.annotation_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        image_size=args.image_size,
        sample_stride=args.sample_stride,
        max_videos=args.max_videos,
        split=args.split,
    )
    set_seed(cfg.seed)
    model, _ = make_model_and_optimizer(cfg)
    metrics = evaluate_one_step(model, cfg, num_batches=args.num_batches)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

if __name__ == "__main__":
    main()
