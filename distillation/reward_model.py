"""Reward modeling utilities for trajectory scoring.

TODO(phase4): replace heuristic placeholders with learned reward model.
"""

from __future__ import annotations

from typing import Any, Dict

from .teacher_interface import TeacherInterface


def score_with_teacher(teacher: TeacherInterface, trajectory: Any) -> Dict[str, float]:
    """Score a trajectory using teacher-provided semantics."""
    return teacher.score_trajectory(trajectory)
