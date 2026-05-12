"""LeRobot LIBERO simulation backend: camera + arm adapters.

LeRobot and LIBERO are lazy-imported so this module imports without them installed.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .camera_base import SimCameraBase, _frame_to_uint8, _make_placeholder_jpeg

logger = logging.getLogger(__name__)

_STEPS_PER_CALL_DEFAULT = 5


def _require_lerobot_libero() -> None:
    missing = []
    try:
        import lerobot.envs.factory  # noqa: F401
    except ImportError:
        missing.append("lerobot")
    try:
        import libero.libero  # noqa: F401
    except ImportError:
        missing.append("libero")
    if missing:
        raise RuntimeError(
            f"LIBERO backend requires: {', '.join(missing)}. "
            "Install on Linux with: pip install -r requirements-libero.txt"
        )


def _unbatch_obs(obs: Any) -> Any:
    import numpy as np

    if obs is None:
        return None
    if isinstance(obs, dict):
        return {k: _unbatch_obs(v) for k, v in obs.items()}
    arr = np.asarray(obs)
    if arr.dtype == object:
        return obs
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return obs


def _first_env_info(info: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    if not isinstance(info, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray) and v.size >= 1:
            out[k] = v.flat[0]
        elif isinstance(v, (list, tuple)) and len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


def _agentview_rgb_from_obs(obs: Any):
    import numpy as np

    if not isinstance(obs, dict):
        return None
    pixels = obs.get("pixels")
    if isinstance(pixels, dict):
        for key in ("image", "agentview_image"):
            img = pixels.get(key)
            if img is not None:
                return _frame_to_uint8(img)
    img = obs.get("agentview_image")
    if img is not None:
        return _frame_to_uint8(img)
    return None


def _eef_pos_from_obs(obs: Any) -> list[float] | None:
    if not isinstance(obs, dict):
        return None
    rs = obs.get("robot_state")
    if not isinstance(rs, dict):
        return None
    eef = rs.get("eef")
    if not isinstance(eef, dict):
        return None
    pos = eef.get("pos")
    if pos is None:
        return None
    import numpy as np

    flat = np.asarray(pos).astype(np.float64).flatten()
    return flat[:3].tolist() if flat.size >= 3 else None


def create_env(
    task_suite: str | None = None,
    task_idx: int | None = None,
    control_mode: str = "relative",
    obs_type: str = "pixels_agent_pos",
):
    """Create a LIBERO task env via LeRobot (SyncVectorEnv, n_envs=1)."""
    _require_lerobot_libero()
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env

    task_suite = (
        task_suite or os.environ.get("LIBERO_TASK_SUITE", "libero_object")
    ).strip()
    if task_idx is None:
        raw = os.environ.get("LIBERO_TASK_IDX", "0").strip()
        try:
            task_idx = int(raw, 10)
        except ValueError:
            logger.warning("Invalid LIBERO_TASK_IDX=%r — using 0", raw)
            task_idx = 0

    obs_type = os.environ.get("LIBERO_OBS_TYPE", "").strip() or obs_type
    control_mode = os.environ.get("LIBERO_CONTROL_MODE", "").strip() or control_mode
    cam = os.environ.get("LIBERO_CAMERA_NAMES", "").strip()
    camera_name = cam or "agentview_image,robot0_eye_in_hand_image"

    oh = int(os.environ.get("LIBERO_OBSERVATION_HEIGHT", "360"))
    ow = int(os.environ.get("LIBERO_OBSERVATION_WIDTH", "360"))

    cfg = LiberoEnv(
        task=task_suite,
        task_ids=[task_idx],
        obs_type=obs_type,
        control_mode=control_mode,
        camera_name=camera_name,
        observation_height=oh,
        observation_width=ow,
    )
    env_dict = make_env(cfg, n_envs=1)
    suite_key = next(iter(env_dict))
    vec = env_dict[suite_key][task_idx]

    sim_seed_raw = os.environ.get("ROBOCREW_SIM_SEED", "").strip()
    if not sim_seed_raw:
        seed = 0
    else:
        try:
            seed = int(sim_seed_raw, 10)
        except ValueError:
            logger.warning("Invalid ROBOCREW_SIM_SEED=%r — using 0", sim_seed_raw)
            seed = 0

    try:
        obs, _info = vec.reset(seed=seed)
    except TypeError:
        obs, _info = vec.reset(seed=[seed])
    setattr(vec, "_longor_task_suite", task_suite)
    setattr(vec, "_longor_task_idx", task_idx)
    setattr(vec, "_longor_seed", seed)
    logger.info(
        "LIBERO env ready: suite=%s task_idx=%s obs_type=%s seed=%s",
        task_suite,
        task_idx,
        obs_type,
        seed,
    )
    return vec


class LiberoCamera(SimCameraBase):
    """RGB from ``pixels['image']`` (agentview); JPEG bytes for VLA."""

    _PLACEHOLDER_TEXT = "libero — no rgb in obs"

    def __init__(self, env, arm=None) -> None:
        super().__init__(env, arm)
        self._placeholder_warned = False

    def _log_vision_degraded(self, reason: str) -> None:
        logger.error(
            "Vision degraded for LIBERO (%s). VLA images may be unavailable.",
            reason,
        )

    def capture_image(self) -> bytes:
        import cv2
        import numpy as np

        self.last_capture_was_placeholder = False
        capture_path = "obs-dict"
        frame = None
        if self._arm is not None:
            obs = getattr(self._arm, "last_obs", None)
            frame = _agentview_rgb_from_obs(_unbatch_obs(obs))
            if frame is not None:
                logger.debug("LiberoCamera: using obs agentview RGB")

        if frame is None:
            capture_path = "render"
            try:
                rgb = self._env.render()
                frame = _frame_to_uint8(rgb) if rgb is not None else None
            except Exception as exc:
                logger.debug("LiberoCamera: render() failed: %s", exc)
                frame = None

        if frame is None:
            if not self._placeholder_warned:
                logger.error("LiberoCamera: no RGB — placeholder JPEG")
                self._placeholder_warned = True
            self.last_capture_was_placeholder = True
            self._mark_vision_degraded("placeholder_capture")
            return _make_placeholder_jpeg(self._PLACEHOLDER_TEXT)

        if not self._first_capture_path_logged:
            logger.info("LiberoCamera: capture path=%s", capture_path)
            self._first_capture_path_logged = True

        rgb = _frame_to_uint8(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        self.vision_health.mark_real(capture_path)
        return buf.tobytes()

class LiberoArm:
    """OSC pose deltas: 3 position + 3 orientation + gripper (LeRobot Box 7)."""

    _STEPS_PER_CALL = _STEPS_PER_CALL_DEFAULT

    def __init__(self, env) -> None:
        import numpy as np

        self._env = env
        space = env.single_action_space
        shape = getattr(space, "shape", None)
        if not shape or shape[0] != 7:
            raise RuntimeError(
                f"LIBERO env expected 7-dim action space; got shape={shape}"
            )
        self._act_dim = int(shape[0])
        self._zero_act = np.zeros((1, self._act_dim), dtype=np.float32)
        self._gripper_value: float = 0.0
        scale = float(os.environ.get("LIBERO_EE_DELTA_SCALE", "25.0"))
        self._pos_scale = scale
        self.last_info: dict[str, Any] = {}
        self.last_tcp_pos: list[float] | None = None
        self.last_obj_pos: list[float] | None = None
        self.last_obs: Any = None
        obs, _r, _t, _tr, info = env.step(self._zero_act)
        self.last_obs = _unbatch_obs(obs)
        self.last_info = _first_env_info(info) if isinstance(info, dict) else {}
        self._update_positions_from_obs(self.last_obs)
        logger.debug("LiberoArm: pos_scale=%.3f", self._pos_scale)

    def _update_positions_from_obs(self, obs: Any) -> None:
        u = _unbatch_obs(obs)
        self.last_tcp_pos = _eef_pos_from_obs(u)
        self.last_obj_pos = None

    def move_ee(self, dx: float, dy: float, dz: float, steps: int | None = None) -> None:
        import numpy as np

        act = self._zero_act.copy()
        d = np.clip(
            self._pos_scale * np.array([dx, dy, dz], dtype=np.float32),
            -1.0,
            1.0,
        )
        act[0, :3] = d
        act[0, 3:6] = 0.0
        act[0, 6] = self._gripper_value
        self._rollout(act, steps=steps or self._STEPS_PER_CALL)

    def set_gripper(self, state: str, steps: int | None = None) -> None:
        if state not in ("open", "close"):
            raise ValueError(
                f"set_gripper: state must be 'open' or 'close', got {state!r}"
            )
        import numpy as np

        self._gripper_value = 1.0 if state == "open" else -1.0
        act = self._zero_act.copy()
        act[0, :6] = 0.0
        act[0, 6] = self._gripper_value
        self._rollout(act, steps=steps or self._STEPS_PER_CALL)

    def _rollout(self, act: Any, steps: int) -> None:
        import numpy as np

        a = np.asarray(act, dtype=np.float32)
        if a.shape == (self._act_dim,):
            a = a.reshape(1, -1)
        for _ in range(steps):
            obs, _r, _t, _tr, info = self._env.step(a)
            self.last_obs = _unbatch_obs(obs)
            self.last_info = (
                _first_env_info(info) if isinstance(info, dict) else {}
            )
            self._update_positions_from_obs(self.last_obs)

    def step(self, action_override=None, steps: int = 1) -> None:
        import numpy as np

        if action_override is None:
            a = self._zero_act
        else:
            arr = np.asarray(action_override, dtype=np.float32).reshape(-1)
            if arr.shape[0] == self._act_dim:
                a = arr.reshape(1, -1)
            else:
                raise ValueError(
                    f"action_override must have shape ({self._act_dim},); got {arr.shape}"
                )
        self._rollout(a, steps=steps)

    def reset_to_initial(self) -> None:
        logger.warning("LiberoArm.reset_to_initial() not implemented for LIBERO")
