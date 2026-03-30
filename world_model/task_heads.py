"""Optional prediction heads for downstream tasks.

These heads are extension points used by later phases for robotics and policy learning.

Example:
    >>> import torch
    >>> from world_model.task_heads import TaskHeads
    >>> heads = TaskHeads(latent_dim=128, action_dim=4)
    >>> out = heads(torch.randn(2, 8, 128))
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class TaskHeads(nn.Module):
    """Lightweight heads for action and value prediction from latent states."""

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, latent_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict auxiliary targets.

        Args:
            latent_seq: Tensor with shape `[B, T, D]`.
        """
        return {
            "pred_actions": self.action_head(latent_seq),
            "pred_values": self.value_head(latent_seq),
        }
