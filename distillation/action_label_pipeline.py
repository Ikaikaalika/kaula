"""Action label pipeline for teacher-enriched supervision."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .teacher_interface import TeacherInterface


def generate_action_targets(
    teacher: TeacherInterface,
    frames: Sequence[Any],
    actions: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Generate action, goal, and object relation targets."""
    payload = teacher.annotate_clip(frames=frames, actions=actions)
    return {
        "action_labels": payload["action_labels"],
        "goal_predictions": payload["goal_predictions"],
        "object_relationships": payload["object_relationships"],
        "teacher_model": payload["teacher_model"],
    }
