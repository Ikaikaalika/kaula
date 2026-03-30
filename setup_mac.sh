#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Core environment ready."
echo "Run toy demo with:"
echo "  python run_demo.py --dataset toy"
echo ""
echo "For DROID RLDS support also run:"
echo "  pip install -r requirements-datasets.txt"
