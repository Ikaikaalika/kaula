"""Teacher model interface for offline distillation data generation.

Teacher usage policy:
- allowed: synthetic labels, semantic supervision, trajectory scoring, dataset enrichment
- forbidden: runtime inference graph dependency

Example:
    >>> teacher = TeacherInterface(model_name="qwen-placeholder")
    >>> labels = teacher.annotate_clip(["frame0", "frame1"], actions=[[0, 1, 0, 0]])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass
class TeacherInterface:
    """Offline teacher API abstraction.

    TODO(phase4): integrate real Qwen/OpenAI-compatible teacher endpoint.
    """

    model_name: str = "qwen-placeholder"

    def annotate_clip(self, frames: Sequence[Any], actions: Sequence[Sequence[float]]) -> Dict[str, Any]:
        """Produce semantic supervision payload for one clip."""
        # Deterministic placeholder annotations for Phase 1/2 scaffolding.
        clip_len = len(frames)
        return {
            "captions": [f"Placeholder caption for frame {i}" for i in range(clip_len)],
            "action_labels": [f"action_step_{i}" for i in range(max(0, clip_len - 1))],
            "goal_predictions": ["continue_trajectory"],
            "object_relationships": ["object_a_near_object_b"],
            "scene_transitions": ["no_major_transition"],
            "teacher_model": self.model_name,
        }

    def score_trajectory(self, latent_or_frame_trajectory: Any) -> Dict[str, float]:
        """Score trajectory quality for optional distillation weighting."""
        return {
            "semantic_consistency": 0.5,
            "task_progress": 0.5,
            "plausibility": 0.5,
        }
