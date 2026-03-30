"""Caption generation pipeline for distillation targets."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .teacher_interface import TeacherInterface


def generate_caption_targets(
    teacher: TeacherInterface,
    frames: Sequence[Any],
    actions: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Generate caption supervision for a clip."""
    payload = teacher.annotate_clip(frames=frames, actions=actions)
    return {
        "captions": payload["captions"],
        "scene_transitions": payload["scene_transitions"],
        "teacher_model": payload["teacher_model"],
    }
