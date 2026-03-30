from __future__ import annotations

from pathlib import Path
import csv
import json

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_metrics(metrics: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

def save_weights(model, out_dir: str | Path, prefix: str = "model") -> Path:
    out_dir = ensure_dir(out_dir)
    out_path = out_dir / f"{prefix}.safetensors"
    model.save_weights(str(out_path))
    return out_path

def save_history_jsonl(rows: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path

def save_history_csv(rows: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("")
        return out_path
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path

def load_weights(model, weight_path: str | Path):
    model.load_weights(str(weight_path))
    return model
