"""RoboCasa backend for OpenDrawer (open and open->close chain tasks).

Linux-only lazy import.

Observation and action layout match GR00T N1.6-3B ``robocasa_panda_omron`` processor
keys: three 256² RGB views, 53-D flattened state, 12-D flattened actions per step.
"""
from __future__ import annotations

import logging
import math
import random
import warnings
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


def to_png(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 frame as lossless PNG bytes."""
    import cv2

    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


OPEN_DRAWER_TASK = "open_drawer"
OPEN_CLOSE_DRAWER_TASK = "open_close_drawer"
OPEN_DRAWER_GYM_ENV_ID = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
_TASK_GYM_ENV_IDS = {OPEN_DRAWER_TASK: OPEN_DRAWER_GYM_ENV_ID,
                     OPEN_CLOSE_DRAWER_TASK: OPEN_DRAWER_GYM_ENV_ID}
_TASK_HORIZONS = {OPEN_DRAWER_TASK: 720, OPEN_CLOSE_DRAWER_TASK: 1000}
_DRAWER_CLOSED_JOINT_MAX = 0.05
_OPEN_CLOSE_SUFFIX = ", then close it"

CAMERA_KEYS = ("res256_image_side_0", "res256_image_side_1",
               "res256_image_wrist_0")
STATE_KEYS = ("gripper_qpos", "base_position", "base_rotation",
              "end_effector_position_relative", "end_effector_rotation_relative",
              "gripper_qvel", "end_effector_position_absolute",
              "end_effector_rotation_absolute", "joint_position",
              "joint_position_cos", "joint_position_sin", "joint_velocity")
STATE_DIMS = (2, 3, 4, 3, 4, 2, 3, 4, 7, 7, 7, 7)
ACTION_KEYS = ("gripper_close", "end_effector_position", "end_effector_rotation",
               "base_motion", "control_mode")
ACTION_DIMS = (1, 3, 3, 4, 1)
IMAGE_SIZE = 256
_EPISODE_CHECKPOINT_VERSION = 2


def expected_state_dim() -> int:
    """Return the flattened proprioceptive vector length for RoboCasa PandaOmron."""
    return sum(STATE_DIMS)


def expected_action_dim() -> int:
    """Return one policy action vector length for RoboCasa PandaOmron."""
    return sum(ACTION_DIMS)


def flatten_state(groups: dict[str, list[float]]) -> list[float]:
    """Concatenate state groups in ``STATE_KEYS`` order."""
    out: list[float] = []
    for key, width in zip(STATE_KEYS, STATE_DIMS):
        chunk = groups.get(key)
        if chunk is None:
            raise ValueError(f"missing state group {key!r}")
        if len(chunk) != width:
            raise ValueError(
                f"state group {key!r} expected {width} floats, got {len(chunk)}")
        out.extend(float(v) for v in chunk)
    return out


def unflatten_state(flat: list[float]) -> dict[str, list[float]]:
    """Split a flat state vector into PandaOmron state group dict entries."""
    if len(flat) != expected_state_dim():
        raise ValueError(
            f"state must have {expected_state_dim()} floats, got {len(flat)}")
    groups: dict[str, list[float]] = {}
    idx = 0
    for key, width in zip(STATE_KEYS, STATE_DIMS):
        groups[key] = [float(v) for v in flat[idx:idx + width]]
        idx += width
    return groups


def unflatten_action(flat: list[float]) -> dict[str, list[float]]:
    """Split one flat action vector into PandaOmron action group dict entries."""
    if len(flat) != expected_action_dim():
        raise ValueError(
            f"action must have {expected_action_dim()} floats, got {len(flat)}")
    groups: dict[str, list[float]] = {}
    idx = 0
    for key, width in zip(ACTION_KEYS, ACTION_DIMS):
        groups[key] = [float(v) for v in flat[idx:idx + width]]
        idx += width
    return groups


def flatten_action_timestep(groups: dict[str, list[float]]) -> list[float]:
    """Concatenate one timestep of action groups in ``ACTION_KEYS`` order."""
    out: list[float] = []
    for key, width in zip(ACTION_KEYS, ACTION_DIMS):
        chunk = groups.get(key)
        if chunk is None:
            raise ValueError(f"missing action group {key!r}")
        if len(chunk) != width:
            raise ValueError(
                f"action group {key!r} expected {width} floats, got {len(chunk)}")
        out.extend(float(v) for v in chunk)
    return out


def validate_finite_vector(values: list[float], label: str) -> None:
    """Raise ValueError when any element is non-finite."""
    for idx, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}[{idx}] is not finite: {value!r}")


def _free_render_contexts_in_own_egl_context() -> None:
    """Make robosuite render contexts free their GL objects in their own context.

    ``MjRenderContext.__del__`` frees MuJoCo's GL objects in whichever EGL context
    is current. A stale context (an old sim kept alive by a reference cycle) is
    finalised whenever the cyclic GC runs, and then deletes the live context's
    same-numbered FBO and textures: every later camera frame is noise or black.
    """
    from mujoco.egl import egl_ext as egl
    from robosuite.renderers.context import egl_context
    from robosuite.utils.binding_utils import MjRenderContext

    original = MjRenderContext.__del__
    if getattr(original, "own_context", False):
        return

    def free_in_own_context(self) -> None:
        live = egl.eglGetCurrentContext()
        own = self.gl_ctx._context
        try:
            self.gl_ctx.make_current()
        except RuntimeError:  # EGL display already terminated at interpreter exit
            own = live
        original(self)  # destroys the context and leaves the thread without one
        if live and live.address != own.address:
            egl.eglMakeCurrent(egl_context.EGL_DISPLAY, egl.EGL_NO_SURFACE,
                               egl.EGL_NO_SURFACE, live)

    free_in_own_context.own_context = True
    MjRenderContext.__del__ = free_in_own_context


def _import_robocasa_gym():
    """Import the Linux-only sim stack and harden robosuite's EGL teardown."""
    try:
        import gymnasium as gym
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*offscreen render.*",
                                    category=UserWarning)
            import robocasa  # noqa: F401
            from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
            import robosuite  # noqa: F401
    except ImportError as exc:
        raise ImportError("RoboCasa is not installed. Follow README "
                          "RoboCasa setup (Linux only).") from exc
    _free_render_contexts_in_own_egl_context()
    return gym


def _resolve_gym_env_id(task: str) -> str:
    """Map a Longor task name to the RoboCasa gym registration id."""
    env_id = _TASK_GYM_ENV_IDS.get(task)
    if env_id is None:
        supported = ", ".join(sorted(_TASK_GYM_ENV_IDS))
        raise RuntimeError(f"RoboCasa backend supports tasks {supported!r} only, "
                           f"got {task!r}")
    return env_id


_TASK_DESCRIPTION_KEY = "annotation.human.action.task_description"


def _task_description_from_reset_obs(reset_obs: object) -> str:
    """Read episode language from the GrootRoboCasaEnv reset observation."""
    if not isinstance(reset_obs, dict):
        raise RuntimeError("RoboCasa reset observation must be a dict with "
                           f"{_TASK_DESCRIPTION_KEY!r}")
    raw = reset_obs.get(_TASK_DESCRIPTION_KEY)
    if raw is None:
        raise RuntimeError(
            f"RoboCasa reset observation missing {_TASK_DESCRIPTION_KEY!r}")
    value = raw
    if isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError(f"RoboCasa {_TASK_DESCRIPTION_KEY!r} is empty")
        value = value[0]
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            raise RuntimeError(f"RoboCasa {_TASK_DESCRIPTION_KEY!r} is empty")
        value = value.flat[0]
    text = str(value).strip()
    if not text:
        raise RuntimeError(
            f"RoboCasa {_TASK_DESCRIPTION_KEY!r} is blank after normalization")
    return text


def _deepcopy_obs(obs: object) -> object:
    """Deep-copy a RoboCasa flat observation dict for checkpoint storage."""
    if isinstance(obs, dict):
        return {str(k): _deepcopy_obs(v) for k, v in obs.items()}
    if isinstance(obs, (list, tuple)):
        return type(obs)(_deepcopy_obs(v) for v in obs)
    if isinstance(obs, np.ndarray):
        return np.array(obs, copy=True)
    return obs


class RoboCasaEnv:
    """One RoboCasa OpenDrawer episode (``open_drawer`` or
    ``open_close_drawer``)."""

    @classmethod
    def from_task(cls, task: str, task_idx: int = 0) -> "RoboCasaEnv":
        """Build one RoboCasa episode.

        ``task_idx`` is reserved for future eval sets.
        """
        if task_idx != 0:
            raise RuntimeError("RoboCasa backend supports task_idx=0 only")
        env = cls.__new__(cls)
        env._init_env(task)
        return env

    def _init_env(self, task: str) -> None:
        env_id = _resolve_gym_env_id(task)
        self._task = task
        self._drawer_was_opened = False
        gym = _import_robocasa_gym()
        self._env = gym.make(env_id, enable_render=True)
        obs, _info = self._env.reset()
        self._last_obs = obs
        reset_desc = _task_description_from_reset_obs(obs)
        if task == OPEN_CLOSE_DRAWER_TASK:
            self.task_description = f"{reset_desc}{_OPEN_CLOSE_SUFFIX}"
        else:
            self.task_description = reset_desc
        self._steps = 0
        self._writer = None
        self._record()
        logger.info("RoboCasa ready: %r - %r", task, self.task_description)

    @property
    def steps(self) -> int:
        """Episode step count so far."""
        return self._steps

    @property
    def horizon(self) -> int:
        """Episode step cap for this task."""
        return _TASK_HORIZONS[self._task]

    def observe(self) -> tuple[tuple[bytes, bytes, bytes], list[float]]:
        """Return policy-ordered PNG views and flattened PandaOmron state."""
        obs = self._last_obs
        if not isinstance(obs, dict):
            raise RuntimeError("RoboCasa observation must be a flat modality dict")
        images: list[bytes] = []
        for cam_key in CAMERA_KEYS:
            images.append(to_png(self._camera_frame(cam_key)))
        state_groups = self._state_groups_from_obs(obs)
        state = flatten_state(state_groups)
        validate_finite_vector(state, "state")
        return (images[0], images[1], images[2]), state

    def _camera_frame(self, cam_key: str) -> np.ndarray:
        """Return one RGB camera frame from the current policy observation."""
        obs = self._last_obs
        if not isinstance(obs, dict):
            raise RuntimeError("RoboCasa observation must be a flat modality dict")
        flat_key = f"video.{cam_key}"
        frame = obs.get(flat_key)
        if frame is None:
            raise RuntimeError(f"RoboCasa observation missing {flat_key!r}")
        arr = np.asarray(frame)
        if arr.ndim == 5:
            arr = arr[0, 0]
        elif arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(
                f"unexpected camera shape for {flat_key}: {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.uint8)

    def _state_groups_from_obs(self,
                               obs: dict[str, object]) -> dict[str, list[float]]:
        groups: dict[str, list[float]] = {}
        for key, width in zip(STATE_KEYS, STATE_DIMS):
            flat_key = f"state.{key}"
            raw = obs.get(flat_key)
            if raw is None:
                raise RuntimeError(f"RoboCasa observation missing {flat_key!r}")
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0, 0]
            elif arr.ndim == 2:
                arr = arr[0]
            vec = arr.reshape(-1)
            if vec.size != width:
                raise RuntimeError(
                    f"state {flat_key} expected {width} floats, got {vec.size}")
            groups[key] = [float(v) for v in vec]
        return groups

    def step(self, action: list[float]) -> bool:
        """Apply one 12-D flat action via sim-native dict conversion."""
        validate_finite_vector(action, "action")
        groups = unflatten_action(action)
        native: dict[str, np.ndarray] = {}
        for key, width in zip(ACTION_KEYS, ACTION_DIMS):
            vec = np.asarray(groups[key], dtype=np.float32).reshape(1, 1, width)
            native[f"action.{key}"] = vec
        obs, _reward, terminated, truncated, _info = self._env.step(native)
        self._last_obs = obs
        self._steps += 1
        if self._task == OPEN_CLOSE_DRAWER_TASK and self._drawer_open_oracle():
            self._drawer_was_opened = True
        self._record()
        return bool(terminated or truncated)

    def _record(self) -> None:
        """Append the primary policy camera frame to the episode video."""
        if not config.SAVE_VIDEO:
            return
        import cv2

        frame = self._camera_frame(CAMERA_KEYS[0])
        if self._writer is None:
            height, width = frame.shape[:2]
            path = Path(config.VIDEO_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(path),
                                           cv2.VideoWriter_fourcc(*"mp4v"),
                                           config.VIDEO_FPS, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def success(self) -> bool:
        """Return whether the episode succeeded (sim task oracle)."""
        if self._task == OPEN_DRAWER_TASK:
            return self._drawer_open_oracle()
        if self._task == OPEN_CLOSE_DRAWER_TASK:
            return self._drawer_was_opened and self._drawer_close_oracle()
        raise RuntimeError(f"unsupported task {self._task!r}")

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Evaluate structured subtask completion (sim oracle)."""
        skill = step["skill"]
        if skill == "open":
            return self._drawer_open_oracle()
        if skill == "close":
            if self._task != OPEN_CLOSE_DRAWER_TASK:
                raise KeyError(
                    f"unsupported skill {skill!r} for task {self._task!r}")
            return self._drawer_close_oracle()
        raise KeyError(f"unsupported skill {skill!r} for task {self._task!r}")

    def _sim_env(self):
        """Return the inner robosuite env (sim ground truth)."""
        return self._env.unwrapped.env

    def drawer_joint_state(self) -> dict[str, float]:
        """Return normalized drawer-door joint values for debug observability."""
        return self._drawer_door_state()

    def _drawer_door_state(self) -> dict[str, float]:
        """Read normalized drawer-door joints from the OpenDrawer fixture."""
        sim_env = self._sim_env()
        drawer = getattr(sim_env, "drawer", None)
        if drawer is None or not hasattr(drawer, "get_door_state"):
            raise RuntimeError(
                "OpenDrawer sim env missing drawer fixture for close oracle")
        raw_state = drawer.get_door_state(env=sim_env)
        if not raw_state:
            raise RuntimeError("drawer close oracle requires non-empty door state")
        return {str(joint): float(value) for joint, value in raw_state.items()}

    def _drawer_close_oracle(self) -> bool:
        """Sim-GT: every drawer door joint closed (joint position <= 0.05)."""
        for joint_p in self._drawer_door_state().values():
            if joint_p > _DRAWER_CLOSED_JOINT_MAX:
                return False
        return True

    def _drawer_open_oracle(self) -> bool:
        """Sim-GT: robosuite OpenDrawer ``_check_success`` (drawer >= 95% open)."""
        return bool(self._env.unwrapped.env._check_success())

    def episode_wrapper_state(self) -> dict[str, int | bool]:
        """Return wrapper counters and latches for paired-start observability."""
        return {"steps": self._steps,
                "drawer_was_opened": self._drawer_was_opened}

    def capture_episode_checkpoint(self) -> dict[str, object]:
        """Capture a post-reset episode snapshot for paired arm replay."""
        if self._steps != 0 or self._drawer_was_opened:
            raise RuntimeError(
                "episode checkpoint must be captured immediately after reset "
                f"(steps={self._steps}, "
                f"drawer_was_opened={self._drawer_was_opened})")
        return {
            "version": _EPISODE_CHECKPOINT_VERSION,
            "task": self._task,
            "task_description": self.task_description,
            "steps": self._steps,
            "drawer_was_opened": self._drawer_was_opened,
            "last_obs": _deepcopy_obs(self._last_obs),
            "sim_state": self._capture_sim_state(),
        }

    def restore_episode_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Rehydrate this episode from a backend-owned checkpoint."""
        self._validate_episode_checkpoint(checkpoint)
        self._reset_backend_internal_state()
        self._restore_sim_state(checkpoint["sim_state"])
        self._last_obs = _deepcopy_obs(checkpoint["last_obs"])
        self.task_description = str(checkpoint["task_description"])
        self._steps = int(checkpoint["steps"])
        self._drawer_was_opened = bool(checkpoint["drawer_was_opened"])
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._sync_sim_after_restore()

    def _validate_episode_checkpoint(self, checkpoint: dict[str, object]) -> None:
        if not isinstance(checkpoint, dict):
            raise RuntimeError("episode checkpoint must be a dict")
        version = checkpoint.get("version")
        if version != _EPISODE_CHECKPOINT_VERSION:
            raise RuntimeError(
                f"unsupported episode checkpoint version {version!r}")
        if checkpoint.get("task") != self._task:
            raise RuntimeError(
                f"checkpoint task {checkpoint.get('task')!r} "
                f"does not match env task {self._task!r}")
        if int(checkpoint.get("steps", -1)) != 0:
            raise RuntimeError("episode checkpoint must be post-reset (steps=0)")
        if bool(checkpoint.get("drawer_was_opened")):
            raise RuntimeError(
                "episode checkpoint must be post-reset (drawer_was_opened=False)")
        for key in ("task_description", "last_obs", "sim_state"):
            if key not in checkpoint:
                raise RuntimeError(f"episode checkpoint missing {key!r}")

    def _capture_backend_state(self) -> dict[str, object]:
        """Capture stepping state that lives outside MuJoCo flattened state."""
        rs_env = self._sim_env()
        backend: dict[str, object] = {}
        if hasattr(rs_env, "timestep"):
            backend["robosuite_timestep"] = int(rs_env.timestep)
        backend["python_random_state"] = random.getstate()
        backend["numpy_random_state"] = np.random.get_state()
        env_rng_states = self._capture_env_rng_states()
        if env_rng_states:
            backend["env_rng_states"] = env_rng_states
        return backend

    def _restore_backend_state(self, backend: object) -> None:
        """Restore backend counters captured by ``_capture_backend_state``."""
        if not isinstance(backend, dict):
            raise RuntimeError(
                "episode checkpoint sim_state.backend must be a dict")
        if "python_random_state" not in backend:
            raise RuntimeError(
                "episode checkpoint sim_state.backend missing "
                "'python_random_state'")
        if "numpy_random_state" not in backend:
            raise RuntimeError(
                "episode checkpoint sim_state.backend missing "
                "'numpy_random_state'")
        rs_env = self._sim_env()
        if "robosuite_timestep" in backend:
            if not hasattr(rs_env, "timestep"):
                raise RuntimeError(
                    "episode checkpoint restore: robosuite_timestep saved but "
                    "robosuite env has no timestep attribute")
            rs_env.timestep = int(backend["robosuite_timestep"])
        random.setstate(backend["python_random_state"])
        np.random.set_state(backend["numpy_random_state"])
        if "env_rng_states" in backend:
            self._restore_env_rng_states(backend["env_rng_states"])

    def _capture_rng_state(self, rng: object) -> dict[str, object] | None:
        """Capture one RNG state in a restoreable form."""
        if rng is None:
            return None
        bit_gen = getattr(rng, "bit_generator", None)
        if bit_gen is not None:
            state = getattr(bit_gen, "state", None)
            if state is not None:
                return {"kind": "np_generator", "state": _deepcopy_obs(state)}
        get_state = getattr(rng, "get_state", None)
        set_state = getattr(rng, "set_state", None)
        if callable(get_state) and callable(set_state):
            return {"kind": "np_random_state", "state": get_state()}
        return None

    def _restore_rng_state(self, rng: object, payload: object,
                           label: str) -> None:
        """Restore one RNG state captured by ``_capture_rng_state``."""
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"episode checkpoint rng payload for {label!r} must be a dict")
        kind = payload.get("kind")
        if kind == "np_generator":
            bit_gen = getattr(rng, "bit_generator", None)
            if bit_gen is None or not hasattr(bit_gen, "state"):
                raise RuntimeError(
                    f"episode checkpoint restore: rng {label!r} "
                    "has no bit_generator.state")
            if "state" not in payload:
                raise RuntimeError(
                    f"episode checkpoint rng payload for {label!r} "
                    "missing 'state'")
            bit_gen.state = _deepcopy_obs(payload["state"])
            return
        if kind == "np_random_state":
            set_state = getattr(rng, "set_state", None)
            if not callable(set_state):
                raise RuntimeError(
                    f"episode checkpoint restore: rng {label!r} "
                    "has no set_state")
            if "state" not in payload:
                raise RuntimeError(
                    f"episode checkpoint rng payload for {label!r} "
                    "missing 'state'")
            set_state(payload["state"])
            return
        raise RuntimeError(
            f"episode checkpoint restore: unsupported rng kind {kind!r} "
            f"for {label!r}")

    def _capture_env_rng_states(self) -> dict[str, dict[str, object]]:
        """Capture env-local RNG states that affect stepping/observations."""
        out: dict[str, dict[str, object]] = {}
        env_roots = {
            "outer_env": self._env,
            "outer_unwrapped": getattr(self._env, "unwrapped", None),
            "sim_env": self._sim_env(),
        }
        for label, env_obj in env_roots.items():
            if env_obj is None:
                continue
            rng_state = self._capture_rng_state(
                getattr(env_obj, "np_random", None))
            if rng_state is not None:
                out[label] = rng_state
        return out

    def _restore_env_rng_states(self, payload: object) -> None:
        """Restore env-local RNG states captured by ``_capture_env_rng_states``."""
        if payload is None:
            return
        if not isinstance(payload, dict):
            raise RuntimeError(
                "episode checkpoint sim_state.backend.env_rng_states must be "
                "a dict")
        env_roots = {
            "outer_env": self._env,
            "outer_unwrapped": getattr(self._env, "unwrapped", None),
            "sim_env": self._sim_env(),
        }
        for label, state_payload in payload.items():
            env_obj = env_roots.get(str(label))
            if env_obj is None:
                raise RuntimeError(
                    f"episode checkpoint restore: unknown env rng scope "
                    f"{label!r}")
            rng = getattr(env_obj, "np_random", None)
            if rng is None:
                raise RuntimeError(
                    f"episode checkpoint restore: env rng scope {label!r} "
                    "has no np_random")
            self._restore_rng_state(rng, state_payload, str(label))

    def _reset_backend_internal_state(self) -> None:
        """Reset backend-owned non-sim buffers before sim-state rehydration."""
        rs_env = self._sim_env()
        outer_reset = getattr(self._env, "reset", None)
        reset_internal = getattr(rs_env, "_reset_internal", None)
        if not callable(outer_reset) and reset_internal is None:
            return
        has_flag = hasattr(rs_env, "deterministic_reset")
        old_flag = getattr(rs_env, "deterministic_reset", None)
        if has_flag:
            rs_env.deterministic_reset = True
        try:
            if callable(outer_reset):
                outer_reset()
            elif reset_internal is not None:
                reset_internal()
        except Exception as exc:
            raise RuntimeError(
                "episode checkpoint restore failed during backend internal "
                "reset") from exc
        finally:
            if has_flag:
                rs_env.deterministic_reset = old_flag

    def _sync_sim_after_restore(self) -> None:
        """Run sim forward and refresh derived state after MuJoCo rehydration."""
        sim = self._sim_env().sim
        forward = getattr(sim, "forward", None)
        if forward is None:
            raise RuntimeError(
                "episode checkpoint restore unsupported: sim.forward missing")
        forward()

    def _capture_sim_state(self) -> dict[str, object]:
        sim = self._sim_env().sim
        get_state = getattr(sim, "get_state", None)
        if get_state is None:
            raise RuntimeError(
                "episode checkpoint capture unsupported: sim.get_state missing")
        state = get_state()
        flatten = getattr(state, "flatten", None)
        set_flat = getattr(sim, "set_state_from_flattened", None)
        if flatten is None or set_flat is None:
            raise RuntimeError(
                "episode checkpoint capture unsupported: sim state flatten API "
                "missing (need MjSimState.flatten and "
                "sim.set_state_from_flattened)")
        flat = np.asarray(flatten(), dtype=np.float64).copy()
        return {
            "encoding": "mjsim_flat_v1",
            "flat": flat,
            "backend": self._capture_backend_state(),
        }

    def _restore_sim_state(self, payload: dict[str, object]) -> None:
        if not isinstance(payload, dict):
            raise RuntimeError("sim_state payload must be a dict")
        encoding = payload.get("encoding")
        if encoding != "mjsim_flat_v1":
            raise RuntimeError(
                f"unsupported sim_state encoding {encoding!r}; "
                f"expected 'mjsim_flat_v1'")
        flat_raw = payload.get("flat")
        if flat_raw is None:
            raise RuntimeError("sim_state payload missing 'flat'")
        saved = np.asarray(flat_raw, dtype=np.float64)
        sim = self._sim_env().sim
        get_state = getattr(sim, "get_state", None)
        set_flat = getattr(sim, "set_state_from_flattened", None)
        if get_state is None or set_flat is None:
            raise RuntimeError(
                "episode checkpoint restore unsupported: sim flatten API missing")
        live_flat = np.asarray(get_state().flatten(), dtype=np.float64)
        if live_flat.shape != saved.shape:
            raise RuntimeError(
                f"episode checkpoint restore: flat state shape {saved.shape} "
                f"does not match live sim {live_flat.shape}; restore on the "
                "same backend instance that captured the checkpoint")
        set_flat(saved)
        self._restore_backend_state(payload.get("backend", {}))

    def close(self) -> None:
        """Release the video writer and close the simulation environment."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("video saved: %s", config.VIDEO_PATH)
        self._env.close()
