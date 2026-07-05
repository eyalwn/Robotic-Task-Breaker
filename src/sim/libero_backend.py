"""LeRobot LIBERO simulation backend: camera + arm adapters.

LeRobot and LIBERO are lazy-imported so this module imports without them installed.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

from .camera_base import SimCameraBase, _frame_to_uint8, _make_placeholder_jpeg
from .video_recorder import VideoRecorder

logger = logging.getLogger(__name__)

_STEPS_PER_CALL_DEFAULT = 5


def _libero_vla_image_flip_enabled() -> bool:
    raw = os.environ.get("LIBERO_VLA_IMAGE_FLIP", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _maybe_flip_libero_vla_frame(frame):
    if _libero_vla_image_flip_enabled():
        return frame[::-1, ::-1]
    return frame


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


def _wrist_rgb_from_obs(obs: Any):
    if not isinstance(obs, dict):
        return None
    pixels = obs.get("pixels")
    if isinstance(pixels, dict):
        for key in ("image2", "robot0_eye_in_hand_image"):
            img = pixels.get(key)
            if img is not None:
                return _frame_to_uint8(img)
    img = obs.get("robot0_eye_in_hand_image")
    if img is not None:
        return _frame_to_uint8(img)
    return None


def _quat2axisangle(quat) -> "np.ndarray":
    """Quaternion (x,y,z,w) → axis-angle; ported from OpenPI libero eval / robosuite."""
    import numpy as np

    q = np.asarray(quat, dtype=np.float64).flatten()
    if q.size < 4:
        return np.zeros(3, dtype=np.float32)
    if q[3] > 1.0:
        q = q.copy()
        q[3] = 1.0
    elif q[3] < -1.0:
        q = q.copy()
        q[3] = -1.0

    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


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


def _gripper_qpos_for_state(grip: "np.ndarray", target_len: int = 2) -> "np.ndarray":
    """Pad or truncate gripper qpos to match OpenPI pi05_libero (2D → 8D state total)."""
    import numpy as np

    if grip.size >= target_len:
        return grip[:target_len].astype(np.float32)
    if grip.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    padded = np.zeros(target_len, dtype=np.float32)
    padded[: grip.size] = grip
    return padded


def _agent_state_from_obs(obs: Any) -> "np.ndarray | None":
    """Extract the 8-dim proprioceptive state from a LIBERO obs dict.

    Matches OpenPI libero eval: concat(eef_pos[3], quat2axisangle(quat)[3], gripper_qpos).
    Priority:
      1. robot_state eef pos (3) + axis-angle (3) + gripper qpos (2, padded if needed)
      2. LeRobot top-level ``agent_pos`` when already 8-dim (legacy)
    Returns None if neither format is available.
    """
    import numpy as np

    if not isinstance(obs, dict):
        return None

    rs = obs.get("robot_state")
    if isinstance(rs, dict):
        gripper = rs.get("gripper") or {}
        eef = rs.get("eef") or {}
        pos = np.asarray(eef.get("pos", []), dtype=np.float32).flatten()
        quat = np.asarray(eef.get("quat", []), dtype=np.float32).flatten()
        grip = np.asarray(gripper.get("qpos", []), dtype=np.float32).flatten()
        if pos.size == 3 and quat.size == 4:
            axis = _quat2axisangle(quat)
            grip_part = _gripper_qpos_for_state(grip, target_len=2)
            return np.concatenate([pos, axis, grip_part])

    agent_pos = obs.get("agent_pos")
    if agent_pos is not None:
        arr = np.asarray(agent_pos, dtype=np.float32).flatten()
        if arr.size == 8:
            return arr

    return None


_HORIZON_ATTRS = ("_max_episode_steps", "horizon", "max_episode_steps")


def _log_episode_horizon(vec: Any) -> None:
    """Discover episode length limit by walking the env wrapper chain."""
    env = vec
    if hasattr(env, "envs") and env.envs:
        env = env.envs[0]

    visited: set[int] = set()
    path_parts: list[str] = []
    for _ in range(16):
        for attr in _HORIZON_ATTRS:
            if hasattr(env, attr):
                value = getattr(env, attr, None)
                if value is not None:
                    source = (
                        ".".join(path_parts + [attr]) if path_parts else attr
                    )
                    logger.info(
                        "LIBERO episode horizon: %s (source=%s)",
                        value,
                        source,
                    )
                    return
        oid = id(env)
        if oid in visited:
            break
        visited.add(oid)
        path_parts.append(type(env).__name__)
        nxt = getattr(env, "unwrapped", None)
        if nxt is not None and nxt is not env:
            env = nxt
            continue
        inner = getattr(env, "env", None) or getattr(env, "_env", None)
        if inner is None:
            break
        env = inner

    logger.info("LIBERO episode horizon: unknown (source=none)")


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
    _log_episode_horizon(vec)
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

        rgb = _maybe_flip_libero_vla_frame(_frame_to_uint8(frame))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        self.vision_health.mark_real(capture_path)
        return buf.tobytes()

    def capture_wrist_image(self) -> bytes:
        import cv2

        frame = None
        if self._arm is not None:
            obs = getattr(self._arm, "last_obs", None)
            frame = _wrist_rgb_from_obs(_unbatch_obs(obs))
            if frame is not None:
                logger.debug("LiberoCamera: using obs wrist RGB")

        if frame is None:
            return _make_placeholder_jpeg("libero — no wrist rgb in obs")

        rgb = _maybe_flip_libero_vla_frame(_frame_to_uint8(frame))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
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
        scale = float(os.environ.get("LIBERO_EE_DELTA_SCALE", "1.0"))
        self._pos_scale = scale
        self.last_ik_debug: dict = {}
        self.last_info: dict[str, Any] = {}
        self.last_tcp_pos: list[float] | None = None
        self.last_obj_pos: list[float] | None = None
        self._target_object: str | None = None
        self.last_obs: Any = None
        self.episode_done = False
        self.env_step_count = 0
        video_path = os.environ.get("LONGOR_VIDEO_OUT", "").strip()
        self._video: VideoRecorder | None = (
            VideoRecorder(video_path) if video_path else None
        )
        obs, _r, _t, _tr, info = env.step(self._zero_act)
        self.env_step_count += 1
        self.last_obs = _unbatch_obs(obs)
        self.last_info = _first_env_info(info) if isinstance(info, dict) else {}
        self._update_positions_from_obs(self.last_obs)
        settle_steps = int(os.environ.get("LIBERO_SETTLE_STEPS", "10"))
        if settle_steps > 0:
            dummy = self._zero_act.copy()
            dummy[0, 6] = -1.0
            self._rollout(dummy, steps=settle_steps)
        logger.debug("LiberoArm: pos_scale=%.3f settle_steps=%d", self._pos_scale, settle_steps)

    def _record(self, obs: Any) -> None:
        if self._video is None:
            return
        try:
            frame = _agentview_rgb_from_obs(_unbatch_obs(obs))
            if frame is None:
                return
            self._video.add_frame(frame)
        except Exception as exc:
            logger.warning("LIBERO video recording disabled: %s", exc)
            self._video = None

    def _update_positions_from_obs(self, obs: Any) -> None:
        u = _unbatch_obs(obs)
        self.last_tcp_pos = _eef_pos_from_obs(u)
        if self._target_object:
            from sim.libero_skills import libero_body_position

            pos = libero_body_position(self._env, self._target_object)
            if pos is not None:
                self.last_obj_pos = pos
        else:
            self.last_obj_pos = None

    def set_target_object(self, body_name: str) -> bool:
        """Track a named MuJoCo body for proximity / min_dist during VLA moves."""
        from sim.libero_skills import libero_body_position

        pos = libero_body_position(self._env, body_name)
        if pos is None:
            self._target_object = None
            self.last_obj_pos = None
            return False
        self._target_object = body_name
        self.last_obj_pos = pos
        return True

    @property
    def agent_state(self) -> "np.ndarray":
        """Current 8-dim proprioceptive state for VLA conditioning.

        Returns zeros if state cannot be extracted (e.g. before first step).
        """
        import numpy as np

        state = _agent_state_from_obs(_unbatch_obs(self.last_obs))
        return state if state is not None else np.zeros(8, dtype=np.float32)

    def move_ee(self, dx: float, dy: float, dz: float, steps: int | None = None) -> None:
        import numpy as np

        self.last_ik_debug = {"cause": "osc_direct"}
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

    def apply_vla_action(
        self,
        dx: float,
        dy: float,
        dz: float,
        drx: float,
        dry: float,
        drz: float,
        grip: float,
        steps: int | None = None,
    ) -> None:
        """Apply a full 7-DOF OSC delta action (OpenPI LIBERO eval path)."""
        import numpy as np

        self.last_ik_debug = {"cause": "osc_direct"}
        act = self._zero_act.copy()
        deltas = np.clip(
            self._pos_scale
            * np.array([dx, dy, dz, drx, dry, drz], dtype=np.float32),
            -1.0,
            1.0,
        )
        act[0, :6] = deltas
        act[0, 6] = float(np.clip(grip, -1.0, 1.0))
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

        if self.episode_done:
            return

        a = np.asarray(act, dtype=np.float32)
        if a.shape == (self._act_dim,):
            a = a.reshape(1, -1)
        for i in range(steps):
            try:
                obs, _r, _t, _tr, info = self._env.step(a)
            except ValueError as exc:
                if "terminated episode" in str(exc):
                    logger.debug("LiberoArm._rollout: env already terminated; stopping early.")
                    self.episode_done = True
                    return
                raise
            self.env_step_count += 1
            self.last_obs = _unbatch_obs(obs)
            self.last_info = (
                _first_env_info(info) if isinstance(info, dict) else {}
            )
            self._update_positions_from_obs(self.last_obs)
            self._record(obs)
            terminated = bool(np.any(_t))
            truncated = bool(np.any(_tr))
            self.episode_done = terminated or truncated
            if self.episode_done:
                raw_success = self.last_info.get("is_success")
                if raw_success is None:
                    raw_success = self.last_info.get("success")
                is_success: bool | None = None
                if raw_success is not None:
                    try:
                        if isinstance(raw_success, np.ndarray):
                            is_success = bool(raw_success.flat[0])
                        else:
                            is_success = bool(raw_success)
                    except Exception:
                        is_success = None
                logger.info(
                    "LiberoArm._rollout: episode ended terminated=%s truncated=%s "
                    "is_success=%s info_keys=%s at env_step_count=%d (%d/%d in this rollout)",
                    terminated,
                    truncated,
                    is_success,
                    list(self.last_info.keys()),
                    self.env_step_count,
                    i + 1,
                    steps,
                )
                break

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

    def _eef_z(self) -> float | None:
        if self.last_tcp_pos is None or len(self.last_tcp_pos) < 3:
            return None
        return float(self.last_tcp_pos[2])

    def _eef_xyz(self) -> list[float] | None:
        if self.last_tcp_pos is None or len(self.last_tcp_pos) < 3:
            return None
        return [float(self.last_tcp_pos[0]), float(self.last_tcp_pos[1]), float(self.last_tcp_pos[2])]

    def _open_gripper_act(self) -> Any:
        act = self._zero_act.copy()
        act[0, 6] = 1.0
        return act

    def _descend_act(self, dz: float) -> Any:
        import numpy as np

        act = self._open_gripper_act()
        act[0, 2] = float(np.clip(-dz * self._pos_scale, -1.0, 1.0))
        return act

    def _xy_retreat_from_basket(self) -> tuple[float, float]:
        """Unit xy direction away from basket (meters), for post-place repositioning."""
        from sim.libero_skills import libero_body_position

        basket = libero_body_position(self._env, "basket_1_main")
        eef = self.last_tcp_pos
        delta = float(os.environ.get("LIBERO_PREPARE_GRAB_XY_RETREAT_DELTA", "0.15"))
        if not basket or not eef:
            return delta, 0.0
        dx, dy = eef[0] - basket[0], eef[1] - basket[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return delta, 0.0
        return (dx / dist) * delta, (dy / dist) * delta

    def _horiz_dist_to_object(self, body_name: str | None) -> float | None:
        if not body_name:
            return None
        from sim.libero_skills import libero_body_position, libero_horizontal_xy_dist

        obj = libero_body_position(self._env, body_name)
        eef = self.last_tcp_pos
        if obj is None or eef is None:
            return None
        return libero_horizontal_xy_dist(eef, obj)

    def _approach_object_xy(self, body_name: str) -> float | None:
        """Open-loop xy steps toward *body_name* at the current eef height."""
        from sim.libero_skills import libero_body_position, libero_horizontal_xy_dist

        max_steps = int(
            os.environ.get(
                "LIBERO_PREPARE_GRAB_APPROACH_MAX_STEPS",
                os.environ.get("LIBERO_PREPARE_GRAB_APPROACH_STEPS", "80"),
            )
        )
        approach_delta = float(
            os.environ.get("LIBERO_PREPARE_GRAB_APPROACH_DELTA", "0.15")
        )
        radius = float(
            os.environ.get("LIBERO_PREPARE_GRAB_APPROACH_RADIUS", "0.15")
        )
        if max_steps <= 0 or approach_delta <= 0.0:
            return self._horiz_dist_to_object(body_name)

        obj = libero_body_position(self._env, body_name)
        if obj is None or self.last_tcp_pos is None:
            return None

        for _ in range(max_steps):
            eef = self.last_tcp_pos
            dist = libero_horizontal_xy_dist(eef, obj)
            if dist < radius:
                break
            dx = obj[0] - eef[0]
            dy = obj[1] - eef[1]
            step_dx = max(-approach_delta, min(approach_delta, dx))
            step_dy = max(-approach_delta, min(approach_delta, dy))
            self.move_ee(step_dx, step_dy, 0.0, steps=1)
            obj = libero_body_position(self._env, body_name)
            if obj is None or self.last_tcp_pos is None:
                break

        return self._horiz_dist_to_object(body_name)

    def prepare_for_grab(self, next_object: str | None = None) -> None:
        """Open gripper, retreat from basket, descend to target_z, then xy-approach."""
        target_z = float(os.environ.get("LIBERO_PREPARE_GRAB_TARGET_Z", "0.55"))
        low_z_max = int(os.environ.get("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "30"))
        descend_steps = int(os.environ.get("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "20"))
        retreat_steps = int(os.environ.get("LIBERO_PREPARE_GRAB_XY_RETREAT_STEPS", "15"))
        approach_steps = int(
            os.environ.get(
                "LIBERO_PREPARE_GRAB_APPROACH_MAX_STEPS",
                os.environ.get("LIBERO_PREPARE_GRAB_APPROACH_STEPS", "80"),
            )
        )
        settle_steps = int(os.environ.get("LIBERO_SETTLE_STEPS", "0"))
        retreat_on = os.environ.get(
            "LIBERO_PREPARE_GRAB_XY_RETREAT", "1"
        ).strip().lower() not in ("0", "false", "no")
        max_steps = (
            1
            + (retreat_steps if retreat_on else 0)
            + max(descend_steps, 0)
            + max(low_z_max, 0)
            + (max(approach_steps, 0) if next_object else 0)
            + max(settle_steps, 0)
        )
        logger.info(
            "prepare_for_grab: start eef_xyz=%s target_z=%s next_object=%s "
            "horiz_dist=%s max_env_steps=%s",
            self._eef_xyz(),
            target_z,
            next_object,
            self._horiz_dist_to_object(next_object),
            max_steps,
        )

        self.set_gripper("open", steps=1)
        self._gripper_value = 1.0

        if retreat_on and retreat_steps > 0:
            rdx, rdy = self._xy_retreat_from_basket()
            if rdx != 0.0 or rdy != 0.0:
                self.move_ee(rdx, rdy, 0.0, steps=retreat_steps)
            logger.info(
                "prepare_for_grab: after_xy_retreat eef_xyz=%s horiz_dist=%s",
                self._eef_xyz(),
                self._horiz_dist_to_object(next_object),
            )

        descend_dz = float(os.environ.get("LIBERO_PREPARE_GRAB_DESCEND_DZ", "0.35"))
        if descend_steps > 0:
            self._rollout(self._descend_act(descend_dz), steps=descend_steps)

        low_z_dz = float(
            os.environ.get("LIBERO_PREPARE_GRAB_LOW_Z_DZ", str(descend_dz))
        )
        low_z_used = 0
        for _ in range(low_z_max):
            z = self._eef_z()
            if z is None or z <= target_z:
                break
            self._rollout(self._descend_act(low_z_dz), steps=1)
            low_z_used += 1

        final_xyz = self._eef_xyz()
        final_z = final_xyz[2] if final_xyz else None
        below = final_z is not None and final_z <= target_z
        logger.info(
            "prepare_for_grab: after_descend eef_xyz=%s target_z=%s "
            "low_z_steps=%s below_target=%s horiz_dist=%s",
            final_xyz,
            target_z,
            low_z_used,
            below,
            self._horiz_dist_to_object(next_object),
        )

        approach_horiz: float | None = None
        if next_object:
            approach_horiz = self._approach_object_xy(next_object)
            logger.info(
                "prepare_for_grab: after_xy_approach eef_xyz=%s horiz_dist=%s",
                self._eef_xyz(),
                approach_horiz,
            )

        logger.info(
            "prepare_for_grab: done eef_xyz=%s target_z=%s horiz_dist=%s",
            self._eef_xyz(),
            target_z,
            approach_horiz if next_object else self._horiz_dist_to_object(next_object),
        )

        if settle_steps > 0:
            self._rollout(self._open_gripper_act(), steps=settle_steps)

    def reset_to_initial(self) -> None:
        logger.warning("LiberoArm.reset_to_initial() not implemented for LIBERO")
