"""Sim-ground-truth perception skills for LIBERO (lazy-imported)."""
from __future__ import annotations

import logging
from typing import Any

from core.skills import DetectedObject, Pose, Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)


def _unwrap_to_sim(vec_env):
    """Return the inner MuJoCo sim object, or None if not reachable."""
    env = vec_env
    if hasattr(env, "envs") and env.envs:
        env = env.envs[0]
    for _ in range(10):
        if hasattr(env, "sim"):
            return env.sim
        inner = getattr(env, "env", None)
        if inner is None:
            break
        env = inner
    return None


def _is_object_body(name: str) -> bool:
    if name in ("worldBody", "world"):
        return False
    prefixes = (
        "robot",
        "gripper",
        "mocap",
        "base",
        "floor",
        "table",
        "wall",
        "ceiling",
    )
    if name.startswith(prefixes):
        return False
    if "_visual" in name or "_collision" in name:
        return False
    return True


class LiberoListObjectsSkill(Skill):
    name = "state.list_objects"
    description = "Return the names of manipulable objects in the current LIBERO scene."
    parameters: list[SkillParam] = []
    returns = "list[str]: object names"

    def __init__(self, vec_env) -> None:
        self._vec_env = vec_env

    def __call__(self, **kwargs: Any) -> list[str]:
        sim = _unwrap_to_sim(self._vec_env)
        if sim is None:
            logger.warning("state.list_objects: could not reach MuJoCo sim")
            return []
        try:
            names = [n for n in sim.model.body_names if _is_object_body(n)]
            return names
        except Exception as exc:
            logger.warning("state.list_objects failed: %s", exc)
            return []


class LiberoDetectSkill(Skill):
    name = "perception.detect"
    description = "Detect a named object in the LIBERO scene and return its pose."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to detect"),
    ]
    returns = "DetectedObject | None"

    def __init__(self, vec_env) -> None:
        self._vec_env = vec_env

    def __call__(self, **kwargs: Any) -> DetectedObject | None:
        import numpy as np

        object_name = str(kwargs.get("object_name", ""))
        sim = _unwrap_to_sim(self._vec_env)
        if sim is None:
            return None
        try:
            body_id = sim.model.body_name2id(object_name)
            pos = sim.data.body_xpos[body_id]
            xyz = np.asarray(pos).flatten()[:3].tolist()
            return DetectedObject(
                name=object_name,
                pose=Pose(x=xyz[0], y=xyz[1], z=xyz[2]),
            )
        except Exception as exc:
            logger.debug("perception.detect(%r) failed: %s", object_name, exc)
            return None


class LiberoLocateSkill(Skill):
    name = "perception.locate"
    description = "Return the 3-D position of a named object in the LIBERO scene."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to locate"),
    ]
    returns = "Pose | None"

    def __init__(self, vec_env) -> None:
        self._detect = LiberoDetectSkill(vec_env)

    def __call__(self, **kwargs: Any) -> Pose | None:
        result = self._detect(object_name=kwargs.get("object_name", ""))
        return result.pose if result is not None else None


def register_libero_skills(registry: SkillRegistry, vec_env) -> None:
    registry.register(LiberoListObjectsSkill(vec_env))
    registry.register(LiberoDetectSkill(vec_env))
    registry.register(LiberoLocateSkill(vec_env))
