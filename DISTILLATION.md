# Distillation

Teacher models are used **offline only** for:
- synthetic labeling
- semantic supervision
- trajectory scoring
- dataset enrichment
- evaluation baselines

Teacher models are **not** part of runtime inference.

## Implemented Components

- `distillation/teacher_interface.py`
- `distillation/caption_pipeline.py`
- `distillation/action_label_pipeline.py`
- `distillation/reward_model.py`
- `training/distill.py`

## Storage Format

Distillation outputs can be stored as:
- JSONL (default)
- parquet (optional)

Example:

```bash
python training/distill.py \
  --model-config configs/model_small.yaml \
  --training-config configs/training.yaml \
  --output outputs/distillation_targets.jsonl
```

## Phase Notes

- Phase 1 uses deterministic placeholder teacher outputs for scaffolding.
- TODO(phase4): integrate real teacher APIs, richer schema validation, and large-scale dataset enrichment jobs.
