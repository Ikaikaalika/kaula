# AGENTS

State-Space Diffusion World Model Project Specification.

## Core Objective

Build a multimodal simulator-grade predictive model:
- long-horizon rollout
- action-conditioned forecasting
- trajectory sampling
- semantic alignment via teacher distillation
- planning-ready latent representations

## Mandatory Architecture

Required:

`video frames -> vision encoder -> latent z_t -> forward latent diffusion -> SSM-conditioned reverse diffusion -> future latent rollout -> decoder/task heads`

Forbidden:
- `encoder -> SSM -> diffusion cleanup -> decoder`
- transformer temporal backbone as core
- diffusion-only temporal core

## Teacher Policy

Teacher model usage is offline-only:
- synthetic labels
- semantic supervision
- trajectory scoring
- dataset enrichment
- evaluation baselines

Teacher model is never part of runtime inference graph.

## Repository Contract

Required directories/files:
- `configs/`
- `datasets/`
- `world_model/`
- `training/`
- `distillation/`
- `visualization/`
- `notebooks/`
- `tests/`
- `README.md`
- `DATASETS.md`
- `ARCHITECTURE.md`
- `DISTILLATION.md`
- `ROADMAP.md`
- `AGENTS.md`

## Training Objective

`L = lambda1 diffusion_loss + lambda2 latent_rollout_loss + lambda3 reconstruction_loss + lambda4 distillation_loss + lambda5 contrastive_alignment_loss`

## Execution Policy

Work in phases.
After each phase:
- update README
- list modified files
- describe assumptions
- describe next steps

Never skip phases.
Always preserve latent state-space diffusion dynamics as temporal core.
