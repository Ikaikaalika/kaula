"""Render decoded rollout trajectories as GIF previews.

Example:
    python visualization/render_rollout.py \
        --input outputs/rollout_sample.npz \
        --output outputs/rollout_preview.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render rollout npz to gif")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/rollout_preview.gif")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def to_uint8(frame_chw: np.ndarray) -> np.ndarray:
    frame = np.transpose(frame_chw, (1, 2, 0))
    frame = np.clip(frame, 0.0, 1.0)
    frame = (frame * 255.0).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    decoded = data["decoded_trajectory"]

    # Shape: [B, S, T, C, H, W]
    b = min(args.batch_index, decoded.shape[0] - 1)
    s = min(args.sample_index, decoded.shape[1] - 1)
    frames = [to_uint8(decoded[b, s, t]) for t in range(decoded.shape[2])]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, format="GIF", duration=1.0 / max(1, args.fps), loop=0)
    print(f"saved rollout gif: {out}")


if __name__ == "__main__":
    main()
