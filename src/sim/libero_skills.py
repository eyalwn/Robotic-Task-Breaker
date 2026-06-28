"""Sim-ground-truth perception skills for LIBERO (lazy-imported)."""
from __future__ import annotations

import logging
import math
import os
from typing import Any

from core.skills import DetectedObject, Pose, Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)

_LIBERO_OBJECT_ALIASES: dict[str, str] = {
    "alphabet_soup_1_main": "alphabet soup can",
    "cream_cheese_1_main": "cream cheese box",
    "basket_1_main": "basket",
}


def _looks_like_sim_body(name: str) -> bool:
    return name.endswith("_main") or "_" in name


def libero_object_human_name(body_name: str) -> str:
    """Map MuJoCo body names to LIBERO natural-language object names."""
    if body_name in _LIBERO_OBJECT_ALIASES:
        return _LIBERO_OBJECT_ALIASES[body_name]
    if _looks_like_sim_body(body_name):
        stem = body_name
        if stem.endswith("_1_main"):
            stem = stem[: -len("_1_main")]
        return stem.replace("_", " ")
    return body_name


def _unwrap_to_sim(vec_env):
    """Return the inner MuJoCo sim object, or None if not reachable."""
    env = vec_env
    # SyncVectorEnv exposes individual envs via .envs
    if hasattr(env, "envs") and env.envs:
        env = env.envs[0]

    # gymnasium .unwrapped jumps straight to the base env
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and unwrapped is not env and hasattr(unwrapped, "sim"):
        return unwrapped.sim

    # Fallback: manual walk via .env / ._env chains
    for _ in range(15):
        if hasattr(env, "sim"):
            return env.sim
        inner = getattr(env, "env", None) or getattr(env, "_env", None)
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


def libero_body_position(vec_env, body_name: str) -> list[float] | None:
    """Return world xyz for a MuJoCo body name, or None if unreachable."""
    import numpy as np

    sim = _unwrap_to_sim(vec_env)
    if sim is None or not body_name:
        return None
    try:
        body_id = sim.model.body_name2id(body_name)
        pos = sim.data.body_xpos[body_id]
        return np.asarray(pos, dtype=np.float64).flatten()[:3].tolist()
    except Exception as exc:
        logger.debug("libero_body_position(%r) failed: %s", body_name, exc)
        return None


def libero_horizontal_xy_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def libero_object_near_basket(
    vec_env,
    object_body: str,
    *,
    basket_body: str = "basket_1_main",
    max_xy: float | None = None,
) -> tuple[bool, float | None]:
    """True when *object_body* xy is within place-smoke distance of the basket."""
    if max_xy is None:
        max_xy = float(os.environ.get("LIBERO_PLACE_SMOKE_MAX_XY", "0.12"))
    obj = libero_body_position(vec_env, object_body)
    basket = libero_body_position(vec_env, basket_body)
    if obj is None or basket is None:
        return False, None
    xy = libero_horizontal_xy_dist(obj, basket)
    return xy <= max_xy, xy


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
        object_name = str(kwargs.get("object_name", ""))
        xyz = libero_body_position(self._vec_env, object_name)
        if xyz is None:
            return None
        return DetectedObject(
            name=object_name,
            pose=Pose(x=xyz[0], y=xyz[1], z=xyz[2]),
        )


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
    registry.register_alias("list_objects", "state.list_objects")
    registry.register(LiberoDetectSkill(vec_env))
    registry.register(LiberoLocateSkill(vec_env))
