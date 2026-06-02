"""Sim-ground-truth perception skills for ManiSkill (lazy-imported)."""
from __future__ import annotations

import logging
from typing import Any

from core.skills import DetectedObject, Pose, Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)

_SKIP_NAMES = frozenset({"agent", "scene", "render_camera"})


def _find_actors(unwrapped) -> dict[str, Any]:
    actors = {}
    for attr, val in unwrapped.__dict__.items():
        if attr.startswith("_") or attr in _SKIP_NAMES:
            continue
        if hasattr(val, "pose") and hasattr(val.pose, "p"):
            actors[attr] = val
    return actors


def _actor_to_pose(actor) -> Pose | None:
    import numpy as np

    try:
        p = actor.pose.p
        arr = np.asarray(p, dtype=np.float64).flatten()
        if arr.size < 3:
            return None
        if arr.ndim == 0:
            return None
        return Pose(
            x=round(float(arr[0]), 6),
            y=round(float(arr[1]), 6),
            z=round(float(arr[2]), 6),
        )
    except Exception:
        return None


def _resolve_actor(actors: dict[str, Any], object_name: str) -> Any | None:
    if object_name in actors:
        return actors[object_name]
    lower = object_name.lower()
    for key, actor in actors.items():
        if key.lower() == lower:
            return actor
    return None


class ManiSkillListObjectsSkill(Skill):
    name = "state.list_objects"
    description = "Return the names of manipulable objects in the current ManiSkill scene."
    parameters: list[SkillParam] = []
    returns = "list[str]: object names"

    def __init__(self, env) -> None:
        self._env = env

    def __call__(self, **kwargs: Any) -> list[str]:
        try:
            unwrapped = self._env.unwrapped
            return sorted(_find_actors(unwrapped).keys())
        except Exception as exc:
            logger.warning("state.list_objects failed: %s", exc)
            return []


class ManiSkillDetectSkill(Skill):
    name = "perception.detect"
    description = "Detect a named object in the ManiSkill scene and return its pose."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to detect"),
    ]
    returns = "DetectedObject | None"

    def __init__(self, env) -> None:
        self._env = env

    def __call__(self, **kwargs: Any) -> DetectedObject | None:
        object_name = str(kwargs.get("object_name", ""))
        try:
            actors = _find_actors(self._env.unwrapped)
            actor = _resolve_actor(actors, object_name)
            if actor is None:
                return None
            pose = _actor_to_pose(actor)
            if pose is None:
                return None
            return DetectedObject(name=object_name, pose=pose)
        except Exception as exc:
            logger.debug("perception.detect(%r) failed: %s", object_name, exc)
            return None


class ManiSkillLocateSkill(Skill):
    name = "perception.locate"
    description = "Return the 3-D position of a named object in the ManiSkill scene."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to locate"),
    ]
    returns = "Pose | None"

    def __init__(self, env) -> None:
        self._detect = ManiSkillDetectSkill(env)

    def __call__(self, **kwargs: Any) -> Pose | None:
        result = self._detect(object_name=kwargs.get("object_name", ""))
        return result.pose if result is not None else None


def register_maniskill_skills(registry: SkillRegistry, env) -> None:
    registry.register(ManiSkillListObjectsSkill(env))
    registry.register(ManiSkillDetectSkill(env))
    registry.register(ManiSkillLocateSkill(env))
