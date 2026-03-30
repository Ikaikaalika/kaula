# Architecture

## Mandatory Temporal Core

This repository implements:

`video -> encoder -> latent z_t -> forward diffusion in latent space -> SSM transition on noisy latent -> reverse diffusion for z_(t+1) -> decoder`

Not implemented:
- `encoder -> SSM -> diffusion cleanup -> decoder`
- pure transformer temporal core
- pure diffusion U-Net temporal core

## Mathematical Form

- Encoder: `z_t = E(x_t)`
- Forward diffusion: `z_t^k = sqrt(alpha_k) z_t^(k-1) + sqrt(1-alpha_k) * eps`
- SSM transition: `h_(t+1) = A h_t + B z_t^k + C a_t`
- Reverse denoise: `z_(t+1)^(k-1) = f_theta(z_(t+1)^k, h_(t+1), k)`
- Decoder: `x_hat_(t+1) = D(z_(t+1)^0)`

## Module Map

- `world_model/encoder.py`: frame -> latent state encoder
- `world_model/latent_diffusion.py`: schedule-aware latent diffusion and denoiser
- `world_model/ssm_diffusion_core.py`: integrated SSM + diffusion temporal operator
- `world_model/decoder.py`: latent -> frame decoder
- `world_model/rollout_sampler.py`: rollout sampling wrapper
- `world_model/task_heads.py`: optional downstream heads
- `world_model/losses.py`: weighted loss stack
- `world_model/schedules.py`: cosine/linear schedules

## Backends

Configured backends:
- `mamba`
- `s4`
- `dss`
- `linear_recurrent`

Phase 1 behavior:
- backends route to linear recurrent fallback with explicit TODO for full backend kernels.
