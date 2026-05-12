"""Best-effort success checker for ManiSkill PickCube-v1."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _to_scalar(v):
    """Convert a tensor / array / scalar to a plain Python float, or return None."""
    if v is None:
        return None
    try:
        # torch.Tensor
        import torch
        if isinstance(v, torch.Tensor):
            return float(v.detach().cpu().flatten()[0])
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return float(v.flat[0])
    except ImportError:
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _obs_debug_summary(obs, max_depth: int = 3, _depth: int = 0) -> str:
    """Recursively summarise obs keys/shapes for a one-time DEBUG dump."""
    indent = "  " * _depth
    if isinstance(obs, dict):
        lines = []
        for k, v in obs.items():
            if _depth < max_depth:
                lines.append(f"{indent}{k}: {_obs_debug_summary(v, max_depth, _depth + 1)}")
            else:
                lines.append(f"{indent}{k}: ...")
        return "{\n" + "\n".join(lines) + f"\n{'  ' * (_depth - 1)}}}" if lines else "{}"
    try:
        import torch
        if isinstance(obs, torch.Tensor):
            return f"Tensor{list(obs.shape)} dtype={obs.dtype}"
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(obs, np.ndarray):
            return f"ndarray{list(obs.shape)} dtype={obs.dtype}"
    except ImportError:
        pass
    if isinstance(obs, (list, tuple)):
        return f"{type(obs).__name__}[{len(obs)}]"
    return repr(obs)


class PickCubeSuccessChecker:
    """Returns True when cube height exceeds a threshold for N consecutive checks.

    Fast path: uses ``env.unwrapped.evaluate()["success"]`` when available
    (ManiSkill3 native flag).  Falls back to parsing the obs dict.
    """

    def __init__(
        self,
        env,
        height_threshold: float = 0.04,
        consecutive_steps: int = 5,
    ) -> None:
        self._env = env
        self._threshold = height_threshold
        self._required = consecutive_steps
        self._streak = 0
        self._warned = False
        self._obs_dumped = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def check(self) -> bool:
        # Fast path: ManiSkill3 evaluate() returns {"success": bool/tensor}
        success = self._check_evaluate()
        if success is not None:
            if success:
                self._streak += 1
            else:
                self._streak = 0
            return self._streak >= self._required

        # Fallback: height-based
        height = self._get_cube_height()
        if height is None:
            return False
        if height > self._threshold:
            self._streak += 1
        else:
            self._streak = 0
        return self._streak >= self._required

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_evaluate(self) -> bool | None:
        """Call env.unwrapped.evaluate() and return the success flag, or None."""
        try:
            raw = getattr(self._env, "unwrapped", self._env)
            info = raw.evaluate()
            if not isinstance(info, dict):
                return None
            raw_success = info.get("success")
            if raw_success is None:
                return None
            scalar = _to_scalar(raw_success)
            if scalar is not None:
                return bool(scalar)
        except Exception:
            pass
        return None

    def _get_cube_height(self) -> float | None:
        try:
            raw = getattr(self._env, "unwrapped", self._env)
            obs = raw.get_obs()
        except Exception:
            obs = None

        if obs is None:
            self._warn_once()
            return None

        if not self._obs_dumped:
            logger.debug("PickCubeSuccessChecker obs structure:\n%s", _obs_debug_summary(obs))
            self._obs_dumped = True

        return self._extract_height(obs)

    def _extract_height(self, obs) -> float | None:
        """Walk known ManiSkill obs layouts to find cube z-position."""
        try:
            if not isinstance(obs, dict):
                self._warn_once()
                return None

            extra = obs.get("extra") or obs.get("obj") or {}
            if not isinstance(extra, dict):
                self._warn_once()
                return None

            # Try common key names for cube / object pose
            for key in ("obj_pose", "cube_pose", "obj_pos", "cube_pos", "goal_pos"):
                v = extra.get(key)
                if v is None:
                    continue
                # Could be a dict with "p" (position) key (sapien.Pose-like)
                if isinstance(v, dict):
                    v = v.get("p", v)
                # Convert tensor/array to plain float
                try:
                    import torch
                    if isinstance(v, torch.Tensor):
                        arr = v.detach().cpu().numpy()
                        # batched shape (1, N) → (N,)
                        if arr.ndim == 2:
                            arr = arr[0]
                        if arr.shape[0] > 2:
                            return float(arr[2])
                        continue
                except ImportError:
                    pass
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        flat = v.flatten()
                        if flat.size > 2:
                            return float(flat[2])
                        continue
                except ImportError:
                    pass
                # Plain sequence
                if isinstance(v, (list, tuple)) and len(v) > 2:
                    return float(v[2])

            self._warn_once()
            return None
        except Exception:
            self._warn_once()
            return None

    def _warn_once(self) -> None:
        if not self._warned:
            logger.warning(
                "Cannot determine cube pose; success detection disabled. "
                "Run with LOG_LEVEL=DEBUG to see obs structure."
            )
            self._warned = True
