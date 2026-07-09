"""LIBERO perception backend selection (sim-GT vs RGB-based real)."""
from __future__ import annotations

import os

from core.skills import SkillRegistry

PERCEPTION_MODES = ("sim_gt", "real")


def resolve_perception_backend() -> str:
    """Return configured perception mode; default sim_gt for M6 compatibility."""
    mode = os.environ.get("PERCEPTION_BACKEND", "sim_gt").strip().lower()
    if mode not in PERCEPTION_MODES:
        raise ValueError(
            f"PERCEPTION_BACKEND must be one of {PERCEPTION_MODES}, got {mode!r}"
        )
    return mode


def register_libero_perception_skills(
    registry: SkillRegistry,
    *,
    vec_env,
    camera,
    mode: str | None = None,
) -> str:
    """Register perception skills for *mode*; returns the resolved mode."""
    resolved = mode if mode is not None else resolve_perception_backend()
    if resolved == "sim_gt":
        from sim.libero_skills import register_libero_skills

        register_libero_skills(registry, vec_env)
        return resolved
    if resolved == "real":
        from sim.libero_real_perception import register_libero_real_perception

        register_libero_real_perception(registry, camera)
        return resolved
    raise ValueError(
        f"PERCEPTION_BACKEND must be one of {PERCEPTION_MODES}, got {resolved!r}"
    )
