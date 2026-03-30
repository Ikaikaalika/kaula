from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

import mlx.core as mx

from world_model.config import WorldModelConfig, apply_overrides, load_config_json
from world_model.data import sample_moving_shapes_batch, set_seed
from world_model.datasets import build_dataset_iterator, describe_supported_datasets
from world_model.train import build_alpha_bars, make_model_and_optimizer, train_epoch
from world_model.evaluate import evaluate_one_step
from world_model.checkpointing import save_history_csv, save_history_jsonl, save_metrics, save_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal MLX world-model demo")
    parser.add_argument("--config", default=None, help="Path to JSON config file (see configs/)")
    parser.add_argument("--dataset", default=None, choices=["toy", "epic_kitchens", "ego4d", "droid"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--save-dir", default="outputs/demo_run")
    return parser.parse_args()


def summarize_epoch(logs: list[dict]) -> dict:
    keys = logs[0].keys()
    return {k: sum(row[k] for row in logs) / float(len(logs)) for k in keys}


def main():
    args = parse_args()
    if args.list_datasets:
        for name, desc in describe_supported_datasets().items():
            print(f"{name}: {desc}")
        return

    base_cfg = load_config_json(args.config) if args.config else WorldModelConfig()
    cfg = apply_overrides(
        base_cfg,
        dataset_name=args.dataset,
        data_root=args.data_root,
        annotation_path=args.annotation_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        image_size=args.image_size,
        num_epochs=args.epochs,
        sample_stride=args.sample_stride,
        max_videos=args.max_videos,
        split=args.split,
    )

    set_seed(cfg.seed)
    _, _, alpha_bars = build_alpha_bars(cfg)
    model, optimizer = make_model_and_optimizer(cfg)

    print(f"Training demo on dataset={cfg.dataset_name!r}...")
    epoch_history = []
    for epoch in range(cfg.num_epochs):
        batch_logs = train_epoch(
            model,
            optimizer,
            cfg,
            alpha_bars,
            num_batches=args.num_batches,
            dataset_iterator=build_dataset_iterator(cfg, num_batches=args.num_batches),
        )
        epoch_summary = {"epoch": epoch, **summarize_epoch(batch_logs)}
        epoch_history.append(epoch_summary)
        print(
            f"epoch={epoch} loss={epoch_summary['loss']:.4f} "
            f"recon={epoch_summary['recon_loss']:.4f} "
            f"latent={epoch_summary['latent_loss']:.4f} "
            f"diff={epoch_summary['diffusion_loss']:.4f}"
        )

    if cfg.dataset_name == "toy":
        frames, actions = sample_moving_shapes_batch(2, cfg.seq_len, cfg.image_size, cfg.channels)
        out = model(frames, actions)
        print("next_frame_pred shape:", out["next_frame_pred"].shape)
    else:
        sample_batch = next(build_dataset_iterator(cfg, num_batches=1))
        out = model(sample_batch["frames"], sample_batch["actions"])
        print("sample frames shape:", sample_batch["frames"].shape)
        print("sample actions shape:", sample_batch["actions"].shape)
        print("next_frame_pred shape:", out["next_frame_pred"].shape)
    metrics = evaluate_one_step(model, cfg, num_batches=min(5, args.num_batches))
    weight_path = save_weights(model, args.save_dir, prefix="final_model")
    history_jsonl = save_history_jsonl(epoch_history, Path(args.save_dir) / "train_history.jsonl")
    history_csv = save_history_csv(epoch_history, Path(args.save_dir) / "train_history.csv")
    save_metrics(metrics, Path(args.save_dir) / "eval_metrics.json")
    print("saved weights to:", weight_path)
    print("saved training history to:", history_jsonl, "and", history_csv)
    print("eval metrics:", metrics)
    print("done")


if __name__ == "__main__":
    main()
