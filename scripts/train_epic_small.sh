#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:-/path/to/EPIC-KITCHENS}"
python run_demo.py --dataset epic_kitchens --data-root "$DATA_ROOT" --epochs 3 --num-batches 20 --batch-size 4 --seq-len 8 --image-size 64
