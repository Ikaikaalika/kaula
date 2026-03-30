#!/usr/bin/env bash
set -euo pipefail
python eval_demo.py --dataset toy --num-batches 10 --batch-size 8 --seq-len 8 --image-size 32
