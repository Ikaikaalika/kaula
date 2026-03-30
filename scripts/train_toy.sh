#!/usr/bin/env bash
set -euo pipefail
python run_demo.py --dataset toy --epochs 3 --num-batches 50 --batch-size 16 --seq-len 8 --image-size 32
