# Experiment Plan

## Main Experiments

### Experiment 1 — One-step latent prediction
Goal: Determine whether the SSM dynamics core predicts future latents well.

Metrics:
- latent MSE
- cosine similarity
- reconstruction loss

### Experiment 2 — Multi-step rollout
Goal: Evaluate rollout quality over increasing horizons.

Metrics:
- rollout latent error
- frame reconstruction drift
- error vs prediction horizon

### Experiment 3 — Stochastic future modeling
Goal: Determine whether latent diffusion improves uncertainty modeling.

Metrics:
- best-of-k latent error
- sample diversity
- qualitative future samples

### Experiment 4 — Efficiency comparison
Goal: Compare SSM and transformer dynamics.

Metrics:
- wall-clock training speed
- memory usage
- throughput
- maximum stable context length

### Experiment 5 — Multimodal joint embedding
Goal: Evaluate whether adding action or language improves the shared latent space.

Metrics:
- latent alignment quality
- downstream prediction quality
- robustness on held-out tasks or scenes

## Baseline Matrix
1. Deterministic latent predictor
2. Transformer world model
3. SSM-only world model
4. Full SSM + latent diffusion model

## Minimum Success Criteria
- End-to-end training on synthetic data
- At least one real dataset integrated and trained
- Baseline vs full model comparison completed
- Results logged in `results/metrics_template.csv`
