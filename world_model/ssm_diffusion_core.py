"""Integrated state-space diffusion temporal operator.

This module is the mandatory temporal core:
- latent forward diffusion is applied to current state
- SSM transition consumes noisy latent + action
- reverse diffusion predicts next latent conditioned on SSM hidden state

It does NOT implement `encoder -> SSM -> diffusion cleanup`.

Example:
    >>> import torch
    >>> from world_model.encoder import VisionEncoder
    >>> from world_model.decoder import LatentDecoder
    >>> from world_model.latent_diffusion import LatentDiffusion
    >>> from world_model.ssm_diffusion_core import StateSpaceDiffusionCore, LatentStateSpaceDiffusionWorldModel
    >>> enc = VisionEncoder(3, 64, 128)
    >>> dec = LatentDecoder(3, 64, 128)
    >>> diff = LatentDiffusion(128, 256, num_steps=16)
    >>> core = StateSpaceDiffusionCore(128, action_dim=4, hidden_dim=256)
    >>> model = LatentStateSpaceDiffusionWorldModel(enc, core, diff, dec)
    >>> frames = torch.randn(2, 8, 3, 64, 64)
    >>> actions = torch.randn(2, 7, 4)
    >>> out = model.forward_train(frames, actions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .decoder import LatentDecoder
from .encoder import VisionEncoder
from .latent_diffusion import LatentDiffusion

SUPPORTED_BACKENDS = {"mamba", "s4", "dss", "linear_recurrent"}


@dataclass
class CoreState:
    """Persistent hidden state for rollout and training loops."""

    hidden: torch.Tensor


class _LinearRecurrentCell(nn.Module):
    """Linear recurrent fallback SSM used in Phase 1.

    TODO(phase2): add true Mamba/S4/DSS backends.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(latent_dim + action_dim + hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, noisy_latent: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy_latent, action, hidden], dim=-1)
        h = F.silu(self.in_proj(x))
        gate = torch.sigmoid(self.gate_proj(h))
        update = torch.tanh(self.update_proj(h))
        return gate * hidden + (1.0 - gate) * update


class StateSpaceDiffusionCore(nn.Module):
    """SSM core that consumes noised latents and actions."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        backend: str = "linear_recurrent",
    ) -> None:
        super().__init__()
        name = backend.lower().strip()
        if name not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Expected one of {sorted(SUPPORTED_BACKENDS)}")
        self.backend = name
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Phase 1 uses linear recurrent fallback for all backend names.
        self.cell = _LinearRecurrentCell(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

    def init_state(self, batch_size: int, device: torch.device) -> CoreState:
        return CoreState(hidden=torch.zeros(batch_size, self.hidden_dim, device=device))

    def transition(
        self,
        noisy_latent_t: torch.Tensor,
        action_t: torch.Tensor,
        state: CoreState,
    ) -> tuple[CoreState, torch.Tensor]:
        hidden_next = self.cell(noisy_latent_t, action_t, state.hidden)
        latent_prior = self.hidden_to_latent(hidden_next)
        return CoreState(hidden=hidden_next), latent_prior


class LatentStateSpaceDiffusionWorldModel(nn.Module):
    """End-to-end latent state-space diffusion world model."""

    def __init__(
        self,
        encoder: VisionEncoder,
        core: StateSpaceDiffusionCore,
        diffusion: LatentDiffusion,
        decoder: LatentDecoder,
        backend_name: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.diffusion = diffusion
        self.decoder = decoder
        self.backend_name = backend_name

    def forward_train(self, frames: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for training losses.

        Args:
            frames: `[B, T, C, H, W]`
            actions: `[B, T-1, A]`
        """
        latents = self.encoder(frames)
        bsz, steps, latent_dim = latents.shape
        if actions.shape[1] != steps - 1:
            raise ValueError(
                f"Expected actions with T-1={steps - 1} steps, got shape {tuple(actions.shape)}"
            )

        state = self.core.init_state(batch_size=bsz, device=frames.device)
        pred_latents = []
        pred_noises = []
        target_noises = []
        hidden_states = []

        for t in range(steps - 1):
            z_t = latents[:, t]
            z_tp1 = latents[:, t + 1]
            a_t = actions[:, t]

            diffusion_t = self.diffusion.sample_timesteps(bsz, device=frames.device)

            input_noise = torch.randn_like(z_t)
            noisy_z_t = self.diffusion.q_sample(z_t, diffusion_t, noise=input_noise)
            state, _ = self.core.transition(noisy_z_t, a_t, state)

            target_noise = torch.randn_like(z_tp1)
            noisy_target = self.diffusion.q_sample(z_tp1, diffusion_t, noise=target_noise)
            pred_noise, _ = self.diffusion.denoiser(noisy_target, z_t, state.hidden, diffusion_t)
            z_tp1_hat = self.diffusion.predict_start_from_noise(noisy_target, diffusion_t, pred_noise)

            hidden_states.append(state.hidden)
            pred_latents.append(z_tp1_hat)
            pred_noises.append(pred_noise)
            target_noises.append(target_noise)

        pred_latents_t = torch.stack(pred_latents, dim=1)
        pred_noises_t = torch.stack(pred_noises, dim=1)
        target_noises_t = torch.stack(target_noises, dim=1)
        hidden_states_t = torch.stack(hidden_states, dim=1)
        decoded = self.decoder(pred_latents_t)

        return {
            "encoded_latents": latents,
            "pred_latents": pred_latents_t,
            "target_latents": latents[:, 1:],
            "pred_noise": pred_noises_t,
            "target_noise": target_noises_t,
            "decoded_frames": decoded,
            "target_frames": frames[:, 1:],
            "hidden_states": hidden_states_t,
        }

    @torch.no_grad()
    def rollout(
        self,
        context_frames: torch.Tensor,
        action_sequence: torch.Tensor,
        rollout_horizon: int,
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Sample future latent and decoded rollouts.

        Args:
            context_frames: `[B, T_ctx, C, H, W]`
            action_sequence: `[B, rollout_horizon, A]`
            rollout_horizon: Number of future steps to sample.
            num_samples: Number of stochastic trajectories.
        """
        if rollout_horizon <= 0:
            raise ValueError(f"rollout_horizon must be > 0, got {rollout_horizon}")
        if action_sequence.shape[1] < rollout_horizon:
            raise ValueError(
                f"action_sequence has {action_sequence.shape[1]} steps, shorter than rollout_horizon={rollout_horizon}"
            )

        latents = self.encoder(context_frames)
        current = latents[:, -1]
        bsz = current.shape[0]
        state = self.core.init_state(batch_size=bsz, device=context_frames.device)

        # Prime hidden state from context trajectory.
        for t in range(latents.shape[1] - 1):
            diffusion_t = torch.zeros((bsz,), device=context_frames.device, dtype=torch.long)
            noisy = self.diffusion.q_sample(latents[:, t], diffusion_t)
            state, _ = self.core.transition(noisy, action_sequence[:, min(t, action_sequence.shape[1] - 1)], state)

        sampled_latents = []
        for step in range(rollout_horizon):
            action_t = action_sequence[:, step]
            diffusion_t = torch.full(
                (bsz,), self.diffusion.num_steps - 1, device=context_frames.device, dtype=torch.long
            )
            noisy_current = self.diffusion.q_sample(current, diffusion_t)
            state, _ = self.core.transition(noisy_current, action_t, state)
            z_next_samples = self.diffusion.reverse_sample(
                hidden_state=state.hidden,
                conditioning_latent=current,
                num_samples=num_samples,
            )
            sampled_latents.append(z_next_samples)
            current = z_next_samples.mean(dim=1)

        latent_rollout = torch.stack(sampled_latents, dim=2)
        decoded_rollout = self.decoder(latent_rollout.reshape(bsz * num_samples, rollout_horizon, self.core.latent_dim))
        decoded_rollout = decoded_rollout.reshape(
            bsz,
            num_samples,
            rollout_horizon,
            decoded_rollout.shape[-3],
            decoded_rollout.shape[-2],
            decoded_rollout.shape[-1],
        )

        uncertainty = latent_rollout.std(dim=1)
        return {
            "latent_trajectory": latent_rollout,
            "decoded_trajectory": decoded_rollout,
            "uncertainty": uncertainty,
        }

    def config_summary(self) -> Dict[str, str]:
        """Human-readable architecture summary."""
        return {
            "temporal_core": "latent_state_space_diffusion",
            "ssm_backend": self.backend_name,
            "diffusion_mode": "inside_temporal_operator",
        }
