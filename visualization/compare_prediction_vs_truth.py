"""Create side-by-side prediction vs. ground-truth video preview.

Example:
    python visualization/compare_prediction_vs_truth.py \
        --prediction outputs/rollout_sample.npz \
        --ground-truth outputs/ground_truth.npz \
        --output outputs/pred_vs_truth.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predicted and true trajectories")
    parser.add_argument("--prediction", required=True, help="npz with decoded_trajectory [B,S,T,C,H,W]")
    parser.add_argument("--ground-truth", required=True, help="npz with frames [B,T,C,H,W] or [T,C,H,W]")
    parser.add_argument("--output", default="outputs/pred_vs_truth.gif")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def _to_rgb(frame_chw: np.ndarray) -> np.ndarray:
    frame = np.transpose(frame_chw, (1, 2, 0))
    frame = np.clip(frame, 0.0, 1.0)
    frame = (frame * 255.0).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, repeats=3, axis=-1)
    return frame


def main() -> None:
    args = parse_args()

    pred_data = np.load(args.prediction)
    pred = pred_data["decoded_trajectory"]
    b = min(args.batch_index, pred.shape[0] - 1)
    s = min(args.sample_index, pred.shape[1] - 1)
    pred_seq = pred[b, s]  # [T,C,H,W]

    gt_data = np.load(args.ground_truth)
    if "frames" not in gt_data:
        raise ValueError("ground-truth npz must contain key 'frames'")
    gt = gt_data["frames"]
    if gt.ndim == 5:
        gt_seq = gt[min(args.batch_index, gt.shape[0] - 1)]
    elif gt.ndim == 4:
        gt_seq = gt
    else:
        raise ValueError(f"Unexpected ground-truth shape: {gt.shape}")

    t_len = min(pred_seq.shape[0], gt_seq.shape[0])
    frames = []
    for t in range(t_len):
        pred_frame = _to_rgb(pred_seq[t])
        gt_frame = _to_rgb(gt_seq[t])
        pad = np.zeros((pred_frame.shape[0], 8, 3), dtype=np.uint8)
        combined = np.concatenate([gt_frame, pad, pred_frame], axis=1)
        frames.append(combined)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames, format="GIF", duration=1.0 / max(1, args.fps), loop=0)
    print(f"saved comparison gif: {out_path}")


if __name__ == "__main__":
    main()
