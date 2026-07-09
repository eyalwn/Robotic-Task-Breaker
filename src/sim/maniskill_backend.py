"""ManiSkill simulation backend: camera + drive adapters.

ManiSkill is lazy-imported so this module can be imported without it installed.
Calling create_env() or any method that needs the sim will raise RuntimeError if
mani_skill / gymnasium are not available.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .camera_base import SimCameraBase, _frame_to_uint8, _make_placeholder_jpeg

warnings.filterwarnings("ignore", category=UserWarning, module="sapien")

logger = logging.getLogger(__name__)

# Steps used to approximate 1 m forward / 1 degree of turn
_STEPS_PER_METER = 20
_STEPS_PER_DEGREE = 1


def _default_sim_backend() -> str | None:
    """Return 'cpu' on Windows (avoids Vulkan DeviceLost); None elsewhere (let ManiSkill pick)."""
    return "cpu" if sys.platform == "win32" else None


def _require_maniskill():
    missing = []
    for pkg in ("mani_skill", "gymnasium"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise RuntimeError(
            f"ManiSkill backend requires: {', '.join(missing)}. "
            "Install with: pip install mani_skill"
        )


def _is_tensor(obj) -> bool:
    try:
        import torch

        return isinstance(obj, torch.Tensor)
    except ImportError:
        return False


def _human_render_mode(env) -> bool:
    rm = getattr(env, "render_mode", None)
    if rm is None:
        u = getattr(env, "unwrapped", None)
        if u is not None:
            rm = getattr(u, "render_mode", None)
    return rm == "human"


def _maybe_refresh_human_viewer(env) -> None:
    """ManiSkill's step() does not call render(); the SAPIEN window stays blank without this."""
    if not _human_render_mode(env):
        return
    try:
        env.render()
    except Exception as exc:
        logger.debug("human viewer refresh skipped: %s", exc)


def create_env(
    env_id: str,
    render_human: bool = False,
    obs_mode: str = "rgbd",
    control_mode: str = "pd_joint_delta_pos",
):
    """Create and reset a ManiSkill gymnasium env. Returns the env object."""
    _require_maniskill()
    import gymnasium as gym

    env_id = env_id.strip()
    if not env_id:
        raise ValueError("env_id is required (e.g. 'PickCube-v1')")

    obs_mode = os.environ.get("MANISKILL_OBS_MODE", "").strip() or obs_mode
    require_rgbd = os.environ.get("ROBOCREW_REQUIRE_RGBD", "1").strip() == "1"
    make_retries = max(1, int(os.environ.get("MANISKILL_GYM_MAKE_RETRIES", "1")))
    retry_sleep = max(
        0.0, float(os.environ.get("MANISKILL_GYM_MAKE_RETRY_SLEEP", "0.35"))
    )
    render_mode = "human" if render_human else "rgb_array"
    render_backend = os.environ.get("MANISKILL_RENDER_BACKEND", "").strip() or None
    sim_backend = os.environ.get("MANISKILL_SIM_BACKEND", "").strip() or _default_sim_backend()
    env_kwargs = {
        "render_mode": render_mode,
        "obs_mode": obs_mode,
        "control_mode": control_mode,
    }
    if render_backend is not None:
        env_kwargs["render_backend"] = render_backend
    if sim_backend is not None:
        env_kwargs["sim_backend"] = sim_backend
    if render_human:
        logger.info("Opening live render window for env: %s", env_id)

    env = None
    rgbd_exc = None
    for attempt in range(1, make_retries + 1):
        try:
            env = gym.make(env_id, **env_kwargs)
            break
        except RuntimeError as exc:
            rgbd_exc = exc
            if obs_mode == "state":
                raise
            is_last = attempt >= make_retries
            if not is_last:
                logger.warning(
                    "gym.make obs_mode=%r attempt %d/%d failed (%s) — retrying.",
                    obs_mode,
                    attempt,
                    make_retries,
                    exc,
                )
                if retry_sleep > 0.0:
                    time.sleep(retry_sleep)
                continue
            if require_rgbd:
                raise RuntimeError(
                    "ManiSkill rgbd observation is required but env creation failed. "
                    f"env={env_id} render_mode={render_mode} "
                    f"render_backend={render_backend or 'default'} "
                    f"obs_mode={obs_mode} retries={make_retries}. "
                    "Set DISPLAY_ENV=0, close Vulkan-heavy apps, update GPU drivers, "
                    "or adjust MANISKILL_RENDER_BACKEND."
                ) from exc
            logger.warning(
                "gym.make obs_mode=%r failed after %d attempts (%s) — "
                "falling back to obs_mode='state'. "
                "This disables obs RGB and may make VLA vision unavailable.",
                obs_mode,
                make_retries,
                exc,
            )
    if env is None:
        obs_mode = "state"
        env_kwargs["obs_mode"] = obs_mode
        env = gym.make(env_id, **env_kwargs)
    elif rgbd_exc is not None:
        logger.info("Recovered env creation after retries (obs_mode=%s).", obs_mode)
    sim_seed_raw = os.environ.get("ROBOCREW_SIM_SEED", "").strip()
    if not sim_seed_raw:
        sim_seed = 0
    else:
        try:
            sim_seed = int(sim_seed_raw, 10)
        except ValueError:
            logger.warning(
                "Invalid ROBOCREW_SIM_SEED=%r — using 0", sim_seed_raw
            )
            sim_seed = 0
    env.reset(seed=sim_seed)
    if render_human:
        _maybe_refresh_human_viewer(env)
    logger.info(
        "Simulator: ManiSkill, env=%s (render_mode=%s, obs_mode=%s, seed=%s)",
        env_id,
        render_mode,
        obs_mode,
        sim_seed,
    )
    logger.debug("Renderer backend: %s", render_backend or "default")
    logger.debug("Simulation backend: %s", sim_backend or "default")
    logger.debug("Controller: %s", control_mode)
    setattr(env, "_sim_obs_mode", obs_mode)
    setattr(env, "_sim_render_backend", render_backend or "default")
    return env


def _is_device_lost_error(exc: RuntimeError) -> bool:
    exc_str = str(exc)
    return "DeviceLost" in exc_str or "device lost" in exc_str.lower()


def _obs_to_rgb(obs) -> "np.ndarray | None":
    """Extract an RGB uint8 array from a ManiSkill obs dict.

    Tries common sensor key paths used by ManiSkill envs. Returns None if no
    RGB data is found so callers can fall back gracefully.
    """
    import numpy as np

    if obs is None or not isinstance(obs, dict):
        return None

    def _is_dict(v):
        return isinstance(v, dict)

    def _try_as_rgb(v):
        try:
            arr = np.asarray(v)
            if arr.ndim >= 3 and arr.shape[-1] == 3:
                return _frame_to_uint8(arr)
        except Exception:
            pass
        return None

    # Try known ManiSkill obs dict paths (3 levels: top/cam/rgb)
    for top_key in ("sensor_data", "image"):
        top = obs.get(top_key)
        if not _is_dict(top):
            continue
        for cam_key in ("base_camera", "hand_camera", "front_camera"):
            cam = top.get(cam_key)
            if not _is_dict(cam):
                continue
            rgb_val = cam.get("rgb")
            if rgb_val is not None:
                result = _try_as_rgb(rgb_val)
                if result is not None:
                    return result

    # Generic: walk dict→dict looking for any 3-channel image
    for v in obs.values():
        if _is_dict(v):
            for vv in v.values():
                if _is_dict(vv):
                    for vvv in vv.values():
                        result = _try_as_rgb(vvv)
                        if result is not None:
                            return result
                else:
                    result = _try_as_rgb(vv)
                    if result is not None:
                        return result

    return None


class ManiSkillCamera(SimCameraBase):
    """Wraps a ManiSkill env and exposes capture_image() compatible with LLMAgent.

    Pass *arm* (a :class:`ManiSkillArm`) so the camera can pull the latest
    observation frame when ``env.render()`` returns nothing (e.g. human mode).
    """

    _PLACEHOLDER_TEXT = "human render mode -- no rgb capture"

    def __init__(self, env, arm=None) -> None:
        super().__init__(env, arm)
        self._placeholder_warned = False
        self._render_failed_warned = False
        self._render_device_lost = False

    def _log_vision_degraded(self, reason: str) -> None:
        logger.error(
            "Vision degraded for this simulation session (%s). "
            "VLA policy images are unavailable/unreliable. "
            "Remediation: keep DISPLAY_ENV=0, update GPU drivers, close other "
            "Vulkan-heavy apps, and retry. If ManiSkill/SAPIEN documents a "
            "software/CPU rendering override for Windows, try that as well.",
            reason,
        )

    def _read_render_frame(self, np_module):
        if self._render_device_lost:
            return None
        try:
            frame = self._env.render()
        except RuntimeError as exc:
            if _is_device_lost_error(exc):
                self._render_device_lost = True
                logger.warning(
                    "ManiSkillCamera: GPU device lost (%s) — "
                    "disabling all render calls for this session.",
                    exc,
                )
            elif not self._render_failed_warned:
                logger.warning(
                    "ManiSkillCamera: env.render() failed (%s) — falling back",
                    exc,
                )
                self._render_failed_warned = True
            return None
        if not isinstance(frame, np_module.ndarray) and not _is_tensor(frame):
            return None
        return frame

    def _read_unwrapped_render_frame(self, np_module):
        if self._render_device_lost:
            return None
        try:
            frame = self._env.unwrapped.render(mode="rgb_array")
        except Exception:
            return None
        if not isinstance(frame, np_module.ndarray) and not _is_tensor(frame):
            return None
        return frame

    def capture_image(self) -> bytes:
        """Render the env and return JPEG bytes.

        Capture chain (each tried only if the previous fails):
          1. arm.last_obs RGB — preferred for VLA in rgbd/rgb observation modes
          2. env.render() — works in rgb_array mode if renderer is healthy
          3. unwrapped.render(mode='rgb_array') — legacy fallback
          4. placeholder JPEG — never returns None
        """
        import cv2
        import numpy as np

        self.last_capture_was_placeholder = False
        capture_path = "obs-dict"
        frame = None

        # Preferred path: obs dict from the most recent arm step.
        if self._arm is not None:
            frame = _obs_to_rgb(getattr(self._arm, "last_obs", None))
            if frame is not None:
                logger.debug("ManiSkillCamera: using obs-dict RGB")

        # Fallback 1: renderer path
        if frame is None:
            capture_path = "render"
            frame = self._read_render_frame(np)

        # Fallback 2: ask unwrapped env directly (skip if GPU device was lost — it may deadlock)
        if frame is None:
            capture_path = "unwrapped-render"
            frame = self._read_unwrapped_render_frame(np)

        if frame is None:
            if not self._placeholder_warned:
                logger.error(
                    "Camera capture unavailable: backend=%s obs_mode=%s placeholder=%s",
                    getattr(self._env, "_sim_render_backend", "unknown"),
                    getattr(self._env, "_sim_obs_mode", "unknown"),
                    True,
                )
            if self._render_device_lost:
                self._mark_vision_degraded("vulkan_device_lost")
            if not self._placeholder_warned:
                logger.debug(
                    "ManiSkillCamera: all RGB capture methods failed; "
                    "returning placeholder JPEG."
                )
                self._placeholder_warned = True
            self.last_capture_was_placeholder = True
            self._mark_vision_degraded("placeholder_capture")
            return _make_placeholder_jpeg(self._PLACEHOLDER_TEXT)

        if not self._first_capture_path_logged:
            logger.debug("ManiSkillCamera: first successful capture via '%s'", capture_path)
            logger.info(
                "Camera capture ready: backend=%s obs_mode=%s path=%s placeholder=%s",
                getattr(self._env, "_sim_render_backend", "unknown"),
                getattr(self._env, "_sim_obs_mode", "unknown"),
                capture_path,
                False,
            )
            self._first_capture_path_logged = True

        rgb = _frame_to_uint8(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        self.vision_health.mark_real(capture_path)
        return buf.tobytes()

class ManiSkillArm:
    """Wraps a ManiSkill env and exposes move_ee / set_gripper for arm control.

    Action-space mapping used by this adapter:
      dims 0..n_arm-1 → joint delta (pd_joint_delta_pos controller)
      last dim        → gripper (+1.0 = open, -1.0 = close)

    Requires action_space.shape[0] >= 2. Typical compatible envs:
      PickCube-v1, PushCube-v1, StackCube-v1, LiftPegUpright-v1
      (use the default pd_joint_delta_pos controller).
    """

    _STEPS_PER_CALL = 5
    _IK_GAIN = float(os.environ.get("MANISKILL_IK_GAIN", "5.0"))
    _IK_DAMPING = float(os.environ.get("MANISKILL_IK_DAMPING", "1e-4"))
    _IK_MAX_JOINT_DELTA = float(os.environ.get("MANISKILL_IK_MAX_JOINT_DELTA", "0.25"))

    def __init__(self, env) -> None:
        import numpy as np

        control_mode = getattr(env.unwrapped, "control_mode", None)
        shape = env.action_space.shape
        if not shape or shape[0] < 4:
            raise RuntimeError(
                "env action space incompatible; expected >=4 dims for joint+gripper+extra "
                f"(got shape {shape}). Use PickCube-v1 / PushCube-v1 / StackCube-v1 "
                "with pd_joint_delta_pos controller and at least 3 EE dims + 1 gripper dim."
            )
        if control_mode != "pd_joint_delta_pos":
            logger.warning(
                "ManiSkillArm: control_mode=%s (expected 'pd_joint_delta_pos'); action dims=%d.",
                control_mode,
                shape[0],
            )
        self._env = env
        self._n_act = shape[0]
        # For real ManiSkill envs, treat the last dim as gripper and the rest as arm.
        # For test stubs (FakeEnv without an agent/robot), we use 3 EE dims + 1 gripper
        # and treat any remaining dims as extras.
        if hasattr(env.unwrapped, "agent"):
            self._n_arm = self._n_act - 1
            self._gripper_dim = self._n_act - 1
        else:
            self._n_arm = max(3, self._n_act - 1)
            self._gripper_dim = 3
        self._zero_act = np.zeros(shape)

        self._gripper_value: float = 1.0
        self._jacobian_logged = False
        self.last_ik_debug: dict = {}

        self.last_info: dict = {}
        self.last_tcp_pos: list[float] | None = None
        self.last_obj_pos: list[float] | None = None
        # Seed last_obs with a zero-action step so ManiSkillCamera's
        # obs-dict fallback works from the very first capture call.
        obs, _, _, _, info = env.step(np.zeros(shape))
        _maybe_refresh_human_viewer(env)
        # Store initial joint configuration so callers can reset between tool runs.
        try:
            self._initial_qpos = env.unwrapped.agent.robot.get_qpos()
        except Exception:
            self._initial_qpos = None
        self.last_obs = obs
        self.last_info = info if isinstance(info, dict) else {}
        self.last_tcp_pos = self._try_get_tcp_pos()
        self.last_obj_pos = self._try_get_obj_pos()
        logger.debug(
            "ManiSkillArm: control_mode=%s, action_space=%s, n_arm=%d, gripper dim=%d",
            control_mode,
            shape,
            self._n_arm,
            self._gripper_dim,
        )
        logger.debug("ManiSkillArm: ik_gain=%.1f", self._IK_GAIN)
        logger.debug(
            "ManiSkillArm: ik_damping=%.2e ik_max_joint_delta=%.3f",
            self._IK_DAMPING,
            self._IK_MAX_JOINT_DELTA,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def move_ee(self, dx: float, dy: float, dz: float, steps: int | None = None) -> None:
        """Apply a delta EE translation using Jacobian-based IK.

        ManiSkill's pd_ee_* controllers require native Pinocchio on Windows.
        To keep Windows simulation working, we run the pd_joint_delta_pos
        controller and numerically solve for joint deltas that produce the
        requested TCP (x, y, z) delta.
        """
        import numpy as np

        act = self._zero_act.copy()
        self.last_ik_debug = {
            "cause": "unknown",
            "scaled_mag": float(np.linalg.norm([dx, dy, dz])),
            "ik_gain": self._IK_GAIN,
            "ik_damping": self._IK_DAMPING,
            "ik_max_joint_delta": self._IK_MAX_JOINT_DELTA,
            "clipped_joints": 0,
            "predicted_tcp_mag": 0.0,
        }
        try:
            J = self._compute_tcp_jacobian()
            try:
                cond = float(np.linalg.cond(J @ J.T))
            except Exception:
                cond = float("inf")
            self.last_ik_debug["jacobian_cond"] = cond
            if cond > 1e6:
                logger.warning(
                    "IK ill-conditioned (cond=%.1e) — arm near singularity. "
                    "Skipping move_ee(%.4f, %.4f, %.4f).",
                    cond,
                    dx,
                    dy,
                    dz,
                )
                self.last_ik_debug["cause"] = "ill_conditioned"
                return
            target = np.asarray([dx, dy, dz], dtype=np.float64)
            # Damped least-squares IK improves stability near singularities.
            jjt = J @ J.T
            lam2 = self._IK_DAMPING * self._IK_DAMPING
            inv = np.linalg.inv(jjt + lam2 * np.eye(3))
            delta_q = J.T @ inv @ target
            delta_q *= self._IK_GAIN
            preclip = delta_q.copy()
            max_delta = self._IK_MAX_JOINT_DELTA
            delta_q = np.clip(delta_q, -max_delta, max_delta)
            clipped_mask = np.abs(preclip) > max_delta + 1e-12
            clipped_count = int(np.count_nonzero(clipped_mask))
            pred_tcp = J @ delta_q
            self.last_ik_debug["clipped_joints"] = clipped_count
            self.last_ik_debug["predicted_tcp_mag"] = float(np.linalg.norm(pred_tcp))
            if clipped_count > 0:
                self.last_ik_debug["cause"] = "joint_delta_clip"
            elif float(np.linalg.norm(pred_tcp)) < 0.5 * float(np.linalg.norm(target)):
                self.last_ik_debug["cause"] = "damping_or_jacobian"
            else:
                self.last_ik_debug["cause"] = "ok"
            act[: self._n_arm] = delta_q
        except AttributeError:
            # Fallback for stub/test envs that do not expose a full ManiSkill agent.robot
            # interface: map the requested EE delta directly onto the first 3 action dims.
            ee_dims = min(3, self._n_arm)
            act[:ee_dims] = np.asarray([dx, dy, dz], dtype=np.float64)[:ee_dims]
            self.last_ik_debug["cause"] = "direct_mapping_fallback"
        # EE moves must not implicitly change the gripper; keep gripper dim neutral here.
        act[-1] = 0.0
        self.step(act, steps=steps or self._STEPS_PER_CALL)

    def set_gripper(self, state: str, steps: int | None = None) -> None:
        """Open or close the gripper for *steps* env steps.

        *state* must be ``"open"`` or ``"close"``.
        Convention: last dim = +1.0 for open, -1.0 for close.
        """
        if state not in ("open", "close"):
            raise ValueError(
                f"set_gripper: state must be 'open' or 'close', got {state!r}"
            )
        self._gripper_value = 1.0 if state == "open" else -1.0
        act = self._zero_act.copy()
        act[self._gripper_dim] = self._gripper_value
        self.step(act, steps=steps or self._STEPS_PER_CALL)

    def reset_to_initial(self) -> None:
        """Reset arm joints to the initial configuration and let physics settle."""
        if getattr(self, "_initial_qpos", None) is None:
            logger.warning(
                "ManiSkillArm.reset_to_initial() called but no initial qpos was recorded."
            )
            return
        robot = self._env.unwrapped.agent.robot
        robot.set_qpos(self._initial_qpos)
        # Let physics/contacts settle so downstream tools start from a stable state.
        self.step(steps=3)
        logger.debug("ManiSkillArm: reset to initial qpos")

    def _compute_tcp_jacobian(self):
        """Return (3, n_arm) positional Jacobian via finite differences."""
        import numpy as np

        robot = self._env.unwrapped.agent.robot
        scene = getattr(self._env.unwrapped, "scene", None)
        qpos_raw, qpos_np, qpos_shape = self._snapshot_qpos(robot)
        tcp_orig = self._flatten_xyz(self._env.unwrapped.agent.tcp.pose.p)
        eps = 1e-5
        J = np.zeros((3, self._n_arm), dtype=np.float64)

        for i in range(self._n_arm):
            tcp_pert = self._measure_perturbed_tcp(
                robot=robot,
                scene=scene,
                qpos_np=qpos_np,
                qpos_raw=qpos_raw,
                qpos_shape=qpos_shape,
                joint_idx=i,
                eps=eps,
            )
            J[:, i] = (tcp_pert - tcp_orig) / eps

        robot.set_qpos(self._match_qpos_type(qpos_np, qpos_raw, qpos_shape))
        self._force_fk(scene)

        # Validation: ensure TCP pose is restored after Jacobian computation
        tcp_restored = self._flatten_xyz(self._env.unwrapped.agent.tcp.pose.p)
        delta = float(np.linalg.norm(tcp_restored - tcp_orig))
        if delta > 10 * eps:
            logger.warning(
                "TCP pose not restored after Jacobian computation — FK may be stale "
                "(||delta||=%.3e, eps=%.3e)",
                delta,
                eps,
            )

        if not self._jacobian_logged:
            logger.debug("Jacobian shape=%s (first compute):\n%s", J.shape, J)
            self._jacobian_logged = True

        return J

    def _force_fk(self, scene) -> None:
        """Best-effort forward-kinematics update after qpos changes."""
        if scene is None:
            return
        update_render = getattr(scene, "update_render", None)
        if callable(update_render):
            update_render()
            return
        step = getattr(scene, "step", None)
        if callable(step):
            step()

    def _snapshot_qpos(self, robot):
        qpos_raw = robot.get_qpos()
        qpos_np = self._to_numpy(qpos_raw)
        qpos_shape = qpos_np.shape
        if qpos_np.ndim == 2 and qpos_np.shape[0] == 1:
            qpos_np = qpos_np[0]
        return qpos_raw, qpos_np, qpos_shape

    def _flatten_xyz(self, v):
        arr = self._to_numpy(v)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.flatten()[:3]

    def _measure_perturbed_tcp(
        self,
        robot,
        scene,
        qpos_np,
        qpos_raw,
        qpos_shape,
        joint_idx: int,
        eps: float,
    ):
        qpos_pert = qpos_np.copy()
        qpos_pert[joint_idx] += eps
        robot.set_qpos(self._match_qpos_type(qpos_pert, qpos_raw, qpos_shape))
        self._force_fk(scene)
        return self._flatten_xyz(self._env.unwrapped.agent.tcp.pose.p)

    def _to_numpy(self, v):
        import numpy as np

        try:
            import torch

            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(v)

    def _match_qpos_type(self, qpos_np, qpos_raw, qpos_shape):
        import numpy as np

        qpos_np = np.asarray(qpos_np)
        if len(qpos_shape) == 2 and qpos_shape[0] == 1:
            qpos_np = qpos_np.reshape((1, -1))
        else:
            qpos_np = qpos_np.reshape(qpos_shape)

        try:
            import torch

            if isinstance(qpos_raw, torch.Tensor):
                return torch.as_tensor(
                    qpos_np,
                    dtype=qpos_raw.dtype,
                    device=qpos_raw.device,
                )
        except ImportError:
            pass
        return qpos_np

    def _tensor_xyz(self, v) -> list[float] | None:
        """Convert a tensor/array (possibly batched) to an [x, y, z] float list."""
        import numpy as np

        try:
            import torch

            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
        except ImportError:
            pass
        a = np.asarray(v).flatten()
        return a[:3].tolist() if a.size >= 3 else None

    def _try_get_tcp_pos(self) -> list[float] | None:
        try:
            return self._tensor_xyz(self._env.unwrapped.agent.tcp.pose.p)
        except Exception:
            return None

    def _try_get_obj_pos(self) -> list[float] | None:
        u = self._env.unwrapped
        for attr in ("cube", "obj", "object", "target_object"):
            try:
                return self._tensor_xyz(getattr(u, attr).pose.p)
            except Exception:
                continue
        return None

    def step(self, action_override=None, steps: int = 1) -> None:
        """Call env.step() *steps* times with *action_override* (or zero action).

        Stores the info dict from the last step in ``self.last_info`` and the
        raw observation in ``self.last_obs`` (used by ManiSkillCamera as an
        image fallback in human render mode).
        """
        import numpy as np

        act = (
            self._zero_act.copy()
            if action_override is None
            else np.asarray(action_override)
        )
        for _ in range(steps):
            obs, _reward, _terminated, _truncated, info = self._env.step(act)
            self.last_info = info if isinstance(info, dict) else {}
            self.last_obs = obs
            _maybe_refresh_human_viewer(self._env)
        self.last_tcp_pos = self._try_get_tcp_pos()
        self.last_obj_pos = self._try_get_obj_pos()


class ManiSkillBackend:
    def __init__(
        self,
        env_id: str,
        render_human: bool = False,
        vla_port: int | None = None,
    ) -> None:
        self._apply_env_defaults()
        self._env = create_env(env_id, render_human=render_human)
        self._arm = ManiSkillArm(self._env)
        self._camera = ManiSkillCamera(self._env, arm=self._arm)
        from sim.success_checker import PickCubeSuccessChecker

        self._checker = PickCubeSuccessChecker(self._env, consecutive_steps=1)
        self._vla_port = (
            vla_port if vla_port is not None else int(os.environ.get("VLA_PORT", "8765"))
        )

    @staticmethod
    def _apply_env_defaults() -> None:
        os.environ.setdefault("VLA_AXIS_REMAP", "2,0,1")
        os.environ.setdefault("VLA_AXIS_SIGNS", "1,-1,-1")
        os.environ.setdefault("VLA_SIM_PICKCUBE_GRIPPER_PROFILE", "true")

    def register_skills(self, registry, goal: str) -> None:
        from sim.maniskill_skills import register_maniskill_skills
        from vla import VLAConfig, create_vla_manipulation
        from vla.arm_skills import register_arm_skills

        register_maniskill_skills(registry, self._env)
        cfg = VLAConfig.from_env()
        tool = create_vla_manipulation(
            tool_name="arm_vla",
            tool_description="VLA-driven arm manipulation",
            task_prompt=goal,
            server_address=f"http://localhost:{self._vla_port}",
            arm=self._arm,
            camera=self._camera,
            cfg=cfg,
        )
        vla_callable = lambda task: tool.invoke({"task": task})
        register_arm_skills(registry, vla_callable)

    def capture_image(self) -> bytes:
        return self._camera.capture_image()

    def success(self) -> bool:
        return bool(self._checker.check())

    def close(self) -> None:
        if hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:
                pass


class ManiSkillDrive:
    """Wraps a ManiSkill env and exposes move_linear / turn_in_place."""

    def __init__(self, env) -> None:
        self._env = env
        import numpy as np

        try:
            self._zero_act = np.zeros(env.action_space.shape)
        except Exception:
            act = env.action_space.sample()
            self._zero_act = act * 0

    def _step(self, action=None):
        import numpy as np

        act = self._zero_act.copy() if action is None else action
        self._env.step(act)

    def move_linear(self, distance_m: float) -> None:
        """Approximate forward/backward motion by stepping the env."""
        import numpy as np

        n_steps = max(1, int(abs(distance_m) * _STEPS_PER_METER))
        logger.info("move_linear(%.2f m) — %d steps", distance_m, n_steps)
        act = self._zero_act.copy()
        # Try to set the first action dimension as a proxy for linear velocity
        if act.shape[0] > 0:
            act[0] = 1.0 if distance_m >= 0 else -1.0
        for _ in range(n_steps):
            self._env.step(act)

    def turn_in_place(self, angle_deg: float) -> None:
        """Approximate rotation by stepping the env."""
        import numpy as np

        n_steps = max(1, int(abs(angle_deg) * _STEPS_PER_DEGREE))
        logger.info("turn_in_place(%.1f°) — %d steps", angle_deg, n_steps)
        act = self._zero_act.copy()
        # Use second action dim as proxy for angular velocity (if available)
        if act.shape[0] > 1:
            act[1] = 1.0 if angle_deg >= 0 else -1.0
        for _ in range(n_steps):
            self._env.step(act)
