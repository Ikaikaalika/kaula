# Roadmap

## Phase 1 (Completed)

- Repo scaffold and required file layout
- Config set (`model_small`, `model_medium`, `model_large`, `diffusion`, `training`)
- Toy dataset loader
- Minimal vision encoder
- Integrated latent state-space diffusion core
- Basic training/eval/rollout scripts

## Phase 2

- EPIC-KITCHENS and Ego4D loader hardening
- Rollout sampler stress tests and caching
- Better action extraction semantics for robotics datasets

## Phase 3

- Full reverse diffusion training refinements
- Multi-step latent rollout optimization
- Unified MLX mode under new architecture modules

## Phase 4

- Teacher integration for captions/action labels/goals
- Distillation objective wiring in train loop
- Large-scale dataset enrichment outputs (JSON/parquet)

## Phase 5

- Evaluation metrics: SSIM/LPIPS/FVD full implementations
- Semantic alignment scoring with teacher evaluator
- Multi-trajectory viewer and latent PCA tools

## Phase 6

- Documentation polish
- Diagrams and benchmark templates
- Expanded tests and CI workflow
