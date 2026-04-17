"""Success detection for LeRobot LIBERO envs via last step info."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LiberoSuccessChecker:
    """True when ``is_success`` (or ``success``) is true for N consecutive checks."""

    def __init__(self, arm, consecutive_steps: int = 1) -> None:
        self._arm = arm
        self._required = max(1, consecutive_steps)
        self._streak = 0

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
        return self._streak >= self._required
