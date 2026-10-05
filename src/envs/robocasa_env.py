"""RoboCasa OpenDrawer backend (linux-only lazy import).

Observation and action layout match GR00T N1.6-3B ``robocasa_panda_omron`` processor
keys: three 256² RGB views, 53-D flattened state, 12-D flattened actions per step.
"""
from __future__ import annotations

import logging
import math
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
OPEN_DRAWER_GYM_ENV_ID = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
OPEN_DRAWER_PROMPT = "Open the drawer"

_OPEN_DEBUG_DISTANCE_FALLBACK = 1.0

CAMERA_KEYS = (
    "res256_image_side_0",
    "res256_image_side_1",
    "res256_image_wrist_0",
)
STATE_KEYS = (
    "gripper_qpos",
    "base_position",
    "base_rotation",
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qvel",
    "end_effector_position_absolute",
    "end_effector_rotation_absolute",
    "joint_position",
    "joint_position_cos",
    "joint_position_sin",
    "joint_velocity",
)
STATE_DIMS = (2, 3, 4, 3, 4, 2, 3, 4, 7, 7, 7, 7)
ACTION_KEYS = (
    "gripper_close",
    "end_effector_position",
    "end_effector_rotation",
    "base_motion",
    "control_mode",
)
ACTION_DIMS = (1, 3, 3, 4, 1)
IMAGE_SIZE = 256


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
            raise ValueError(f"state group {key!r} expected {width} floats, got {len(chunk)}")
        out.extend(float(v) for v in chunk)
    return out


def unflatten_state(flat: list[float]) -> dict[str, list[float]]:
    """Split a flat state vector into PandaOmron state group dict entries."""
    if len(flat) != expected_state_dim():
        raise ValueError(
            f"state must have {expected_state_dim()} floats, got {len(flat)}"
        )
    groups: dict[str, list[float]] = {}
    idx = 0
    for key, width in zip(STATE_KEYS, STATE_DIMS):
        groups[key] = [float(v) for v in flat[idx : idx + width]]
        idx += width
    return groups


def unflatten_action(flat: list[float]) -> dict[str, list[float]]:
    """Split one flat action vector into PandaOmron action group dict entries."""
    if len(flat) != expected_action_dim():
        raise ValueError(
            f"action must have {expected_action_dim()} floats, got {len(flat)}"
        )
    groups: dict[str, list[float]] = {}
    idx = 0
    for key, width in zip(ACTION_KEYS, ACTION_DIMS):
        groups[key] = [float(v) for v in flat[idx : idx + width]]
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
            raise ValueError(f"action group {key!r} expected {width} floats, got {len(chunk)}")
        out.extend(float(v) for v in chunk)
    return out


def validate_finite_vector(values: list[float], label: str) -> None:
    """Raise ValueError when any element is non-finite."""
    for idx, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}[{idx}] is not finite: {value!r}")


def _import_robocasa_gym():
    try:
        import gymnasium as gym
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*offscreen render.*",
                category=UserWarning,
            )
            import robocasa  # noqa: F401
            from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
            import robosuite  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "RoboCasa is not installed. Follow README RoboCasa setup (Linux only)."
        ) from exc
    return gym


def _owned_attr(obj: object, name: str) -> object:
    """Read an attribute defined on this object, not via ``__getattr__``.

    Gymnasium 0.29 wrappers forward every missing name (and warn). The
    OpenDrawer task sits under those wrappers, so oracle lookup must ignore
    that forwarding.
    """
    try:
        instance_dict = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        instance_dict = {}
    if name in instance_dict:
        return instance_dict[name]
    for cls in type(obj).__mro__:
        if name not in cls.__dict__:
            continue
        desc = cls.__dict__[name]
        if isinstance(desc, property) or callable(desc):
            return desc.__get__(obj, type(obj))
        return desc
    raise AttributeError(name)


def _iter_owned_layers(env: object):
    """Yield nested env objects via owned ``env`` / ``unwrapped`` links."""
    seen: set[int] = set()
    pending = [env]
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("env", "unwrapped"):
            try:
                nxt = _owned_attr(current, attr)
            except AttributeError:
                continue
            if nxt is not None and id(nxt) not in seen:
                pending.append(nxt)


def _call_check_success(env: object) -> bool:
    """Return the nested RoboCasa task ``_check_success`` result."""
    for layer in _iter_owned_layers(env):
        try:
            check = _owned_attr(layer, "_check_success")
        except AttributeError:
            continue
        if callable(check):
            return bool(check())
    raise RuntimeError(
        "RoboCasa OpenDrawer drawer oracle requires _check_success on the task env"
    )


def _resolve_sim(env: object) -> object:
    """Return the MuJoCo sim owned by the nested robosuite env."""
    for layer in _iter_owned_layers(env):
        try:
            sim = _owned_attr(layer, "sim")
        except AttributeError:
            continue
        if sim is not None and getattr(sim, "data", None) is not None:
            return sim
    raise RuntimeError("RoboCasa drawer oracle requires sim access")


def _task_drawer(env: object) -> object:
    """Return the OpenDrawer fixture on the nested robosuite task."""
    for layer in _iter_owned_layers(env):
        try:
            drawer = _owned_attr(layer, "drawer")
        except AttributeError:
            continue
        if drawer is not None:
            return drawer
    raise KeyError("drawer")


def _resolve_gym_env_id(task: str) -> str:
    """Map a Longor task name to the RoboCasa gym registration id."""
    if task != OPEN_DRAWER_TASK:
        raise RuntimeError(
            f"RoboCasa backend supports task {OPEN_DRAWER_TASK!r} only, got {task!r}"
        )
    return OPEN_DRAWER_GYM_ENV_ID


def _coerce_nonempty_text(raw: object) -> str:
    """Normalize a candidate language payload to a non-empty task string."""
    if raw is None:
        return ""
    value = raw
    if isinstance(value, (tuple, list)):
        if not value:
            return ""
        value = value[0]
    text = str(value).strip()
    return text


def _mapping_text(mapping: object, keys: tuple[str, ...]) -> str:
    """Return the first non-empty string value from a mapping-like object."""
    if not isinstance(mapping, dict):
        return ""
    for key in keys:
        text = _coerce_nonempty_text(mapping.get(key))
        if text:
            return text
    return ""


def _resolve_task_description(
    task: str,
    env: object,
    reset_obs: object,
    reset_info: object,
) -> str:
    """Resolve task language from RoboCasa wrappers with M2 OpenDrawer fallback."""
    info_text = _mapping_text(reset_info, ("language", "task_description", "instruction"))
    if info_text:
        return info_text
    obs_text = _mapping_text(
        reset_obs,
        (
            "annotation.human.action.task_description",
            "language",
            "task_description",
        ),
    )
    if obs_text:
        return obs_text
    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        for key in ("language", "task_description", "instruction"):
            try:
                text = _coerce_nonempty_text(get_wrapper_attr(key))
            except Exception:
                continue
            if text:
                return text
    unwrapped = getattr(env, "unwrapped", None)
    for holder in (unwrapped,):
        if holder is None:
            continue
        for key in ("language", "task_description", "instruction"):
            text = _coerce_nonempty_text(getattr(holder, key, ""))
            if text:
                return text
    if task == OPEN_DRAWER_TASK:
        logger.warning(
            "RoboCasa language missing; using fallback prompt for %r", OPEN_DRAWER_TASK
        )
        return OPEN_DRAWER_PROMPT
    raise RuntimeError(
        "RoboCasa episode language instruction is missing; "
        "refusing to substitute a default task string"
    )


class RoboCasaEnv:
    """One RoboCasa OpenDrawer episode."""

    @classmethod
    def from_task(cls, task: str, task_idx: int = 0) -> "RoboCasaEnv":
        """Build one OpenDrawer episode (``task_idx`` reserved for future eval sets)."""
        if task_idx != 0:
            raise RuntimeError(
                "RoboCasa OpenDrawer vertical slice supports task_idx=0 only"
            )
        env = cls.__new__(cls)
        env._init_env(task)
        return env

    def _init_env(self, task: str) -> None:
        gym = _import_robocasa_gym()
        env_id = _resolve_gym_env_id(task)
        self._env = gym.make(env_id, enable_render=True)
        obs, info = self._env.reset()
        self._last_obs = obs
        self.task_description = _resolve_task_description(task, self._env, obs, info)
        self._steps = 0
        self._writer = None
        self._record()
        logger.info("RoboCasa ready: %r - %r", task, self.task_description)

    @property
    def steps(self) -> int:
        return self._steps

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
            raise RuntimeError(f"unexpected camera shape for {flat_key}: {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.uint8)

    def _state_groups_from_obs(self, obs: dict[str, object]) -> dict[str, list[float]]:
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
                    f"state {flat_key} expected {width} floats, got {vec.size}"
                )
            groups[key] = [float(v) for v in vec]
        return groups

    def _refresh_obs(self) -> object:
        """Re-render all cameras and rebuild the policy obs dict (robosuite caches camera observables)."""
        inner = self._env.unwrapped
        raw = inner.env._get_observations(force_update=True)
        return inner.get_groot_observation(inner.get_basic_observation(raw))

    def step(self, action: list[float]) -> bool:
        """Apply one 12-D flat action via sim-native dict conversion."""
        validate_finite_vector(action, "action")
        groups = unflatten_action(action)
        native: dict[str, np.ndarray] = {}
        for key, width in zip(ACTION_KEYS, ACTION_DIMS):
            vec = np.asarray(groups[key], dtype=np.float32).reshape(1, 1, width)
            native[f"action.{key}"] = vec
        _obs, _reward, terminated, truncated, _info = self._env.step(native)
        # step()'s returned cameras can stay cached; force a full observable resample.
        self._last_obs = self._refresh_obs()
        self._steps += 1
        self._record()
        return bool(terminated or truncated)

    def _record(self) -> None:
        """Append the primary policy camera frame to the episode video."""
        if not config.SAVE_VIDEO:
            return
        # RoboCasa camera frames are valid only when steps % 4 == 0.
        if self._steps % 4:
            return
        import cv2

        frame = self._camera_frame(CAMERA_KEYS[0])
        if self._writer is None:
            height, width = frame.shape[:2]
            path = Path(config.VIDEO_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                config.VIDEO_FPS,
                (width, height),
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def success(self) -> bool:
        """Return whether the OpenDrawer episode succeeded (sim task oracle)."""
        return self._drawer_open_oracle()

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Evaluate structured subtask completion (OpenDrawer sim oracle)."""
        skill = step["skill"]
        if skill == "open":
            return self._drawer_open_oracle()
        raise KeyError(f"unsupported skill: {skill!r}")

    def _drawer_open_oracle(self) -> bool:
        """Sim-GT: nested OpenDrawer ``_check_success`` (drawer >= 95% open)."""
        return _call_check_success(self._env)

    def calc_distances(self, origin_name: str, target_name: str) -> float:
        """Return EEF-to-drawer distance (metres). Fallback if the site is missing."""
        if origin_name != "EEF":
            raise KeyError(f"unsupported origin: {origin_name!r}")
        try:
            groups = self._state_groups_from_obs(self._last_obs)
            eef = np.asarray(
                groups["end_effector_position_absolute"],
                dtype=np.float64,
            )
            sim = _resolve_sim(self._env)
            # Planner text ("right drawer") is not a MuJoCo body name.
            site = f"{_task_drawer(self._env).naming_prefix}ext_p0"
            target = np.asarray(sim.data.get_site_xpos(site), dtype=np.float64)
            return float(np.linalg.norm(eef - target))
        except Exception:
            return _OPEN_DEBUG_DISTANCE_FALLBACK

    def close(self) -> None:
        """Release the video writer and close the simulation environment."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("video saved: %s", config.VIDEO_PATH)
        self._env.close()
