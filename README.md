# Latent State-Space Diffusion World Model

This repository implements a **latent state-space diffusion world model** where diffusion is part of temporal dynamics:

`vision encoder -> latent states -> forward latent diffusion -> SSM-conditioned reverse diffusion -> future latent rollout -> decoder`

## Phase 1 Status

Phase 1 goals:
- repo scaffold
- configs
- toy dataset loader
- minimal encoder
- minimal SSM diffusion block

Phase 1 completion status:
- completed scaffold under required top-level directories
- completed runnable toy-data training/eval/rollout scripts
- completed integrated latent diffusion + SSM temporal core (no post-SSM diffusion cleanup)
- added explicit TODO markers for later phases (dataset depth, distillation integration, advanced metrics)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python training/train.py \
  --model-config configs/model_small.yaml \
  --diffusion-config configs/diffusion.yaml \
  --training-config configs/training.yaml \
  --max-steps 20

python training/rollout.py \
  --checkpoint outputs/phase1_train/model_final.pt \
  --horizon 8 \
  --num-samples 4

python training/evaluate.py \
  --checkpoint outputs/phase1_train/model_final.pt
```

Default dataset root in config: `"/Volumes/Tyler HDD"` (see `configs/training.yaml`).

## Architecture Guardrails

- Diffusion operates in latent dynamics.
- Decoder only receives denoised latent states.
- Runtime inference graph does not include teacher models.
- Temporal operator is state-space diffusion, not transformer-only or diffusion-only.

## Runtime Modes

- `pytorch` mode: implemented in `training/train.py`.
- `mlx` prototype mode: delegated to existing `run_demo.py` for Apple Silicon local prototyping; migration to unified architecture is a Phase 3 TODO.

## Modified Files (Phase 1)

- `configs/model_small.yaml`
- `configs/model_medium.yaml`
- `configs/model_large.yaml`
- `configs/diffusion.yaml`
- `configs/training.yaml`
- `datasets/common.py`
- `datasets/toy_dataset.py`
- `datasets/epic_kitchens.py`
- `datasets/ego4d.py`
- `datasets/droid.py`
- `datasets/bridge_data.py`
- `datasets/something_something_v2.py`
- `datasets/__init__.py`
- `world_model/encoder.py`
- `world_model/latent_diffusion.py`
- `world_model/ssm_diffusion_core.py`
- `world_model/rollout_sampler.py`
- `world_model/decoder.py`
- `world_model/task_heads.py`
- `world_model/losses.py`
- `world_model/schedules.py`
- `world_model/__init__.py`
- `training/config_utils.py`
- `training/model_factory.py`
- `training/train.py`
- `training/evaluate.py`
- `training/rollout.py`
- `training/distill.py`
- `distillation/teacher_interface.py`
- `distillation/caption_pipeline.py`
- `distillation/action_label_pipeline.py`
- `distillation/reward_model.py`
- `visualization/render_rollout.py`
- `visualization/compare_prediction_vs_truth.py`
- `tests/test_schedules.py`
- `tests/test_toy_dataset.py`
- `tests/test_core_forward.py`
- `ARCHITECTURE.md`
- `DATASETS.md`
- `DISTILLATION.md`
- `ROADMAP.md`
- `AGENTS.md`

## Assumptions (Phase 1)

- Phase 1 emphasizes architectural correctness and runnable scaffolding over benchmark-quality metrics.
- Advanced backends (`mamba`, `s4`, `dss`) are config-selectable but currently mapped to linear recurrent fallback with explicit TODO.
- LPIPS/FVD/teacher-semantic eval are scaffolded with placeholders for Phase 5.

## Next Steps

1. Phase 2: deepen EPIC/Ego4D loading + rollout sampler stress tests.
2. Phase 3: full reverse sampler training logic and MLX path migration.
3. Phase 4: teacher-backed distillation pipelines.
4. Phase 5: full evaluation metric stack and visualization tools.
5. Phase 6: docs hardening, diagrams, broader test coverage.
