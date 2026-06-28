"""Success detection for LeRobot LIBERO envs via last step info."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def plan_return_indicates_failure(value: Any) -> bool:
    """True when a plan return literal looks like an explicit failure message."""
    return isinstance(value, str) and "failed to" in value.lower()


class LiberoSuccessChecker:
    """True when ``is_success`` (or ``success``) is true for N consecutive checks.

    When *env* and *task_objects* are set, also requires each object's xy to be
  within ``LIBERO_PLACE_SMOKE_MAX_XY`` of ``basket_1_main`` (place-smoke gate).
    """

    def __init__(
        self,
        arm,
        consecutive_steps: int = 1,
        env=None,
        task_objects: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self._arm = arm
        self._env = env
        self._task_objects = tuple(task_objects) if task_objects else ()
        self._required = max(1, consecutive_steps)
        self._streak = 0

    def refresh(self) -> None:
        """Advance sim one step so ``last_info`` reflects the current scene."""
        step = getattr(self._arm, "step", None)
        if callable(step):
            step()

    def task_objects_geometry(self) -> tuple[bool, dict[str, float | None]]:
        """Return (all_near_basket, per-object xy distance to basket)."""
        if not self._env or not self._task_objects:
            return True, {}
        from sim.libero_skills import libero_object_near_basket

        dists: dict[str, float | None] = {}
        all_near = True
        for name in self._task_objects:
            near, xy = libero_object_near_basket(self._env, name)
            dists[name] = xy
            if not near:
                all_near = False
        return all_near, dists

    def check(self) -> bool:
        info = getattr(self._arm, "last_info", None)
        if not isinstance(info, dict):
            return False
        raw = info.get("is_success")
        if raw is None:
            raw = info.get("success")
        if raw is None:
            return False
        try:
            import numpy as np

            if isinstance(raw, np.ndarray):
                ok = bool(raw.flat[0])
            else:
                ok = bool(raw)
        except Exception:
            logger.debug("LiberoSuccessChecker: could not parse success flag %r", raw)
            return False
        if ok:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak < self._required:
            return False
        geom_ok, _ = self.task_objects_geometry()
        return geom_ok
