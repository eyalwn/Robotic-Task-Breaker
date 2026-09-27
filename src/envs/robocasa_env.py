"""RoboCasa OpenDrawer backend (linux-only lazy import).

Observation and action layout match GR00T N1.6-3B ``robocasa_panda_omron`` processor
keys: three 256² RGB views, 53-D flattened state, 12-D flattened actions per step.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from envs.vlabench_env import to_jpeg

logger = logging.getLogger(__name__)

OPEN_DRAWER_TASK = "open_drawer"
OPEN_DRAWER_GYM_ENV_ID = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"

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
        import robocasa  # noqa: F401
        import robosuite  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "RoboCasa is not installed. Follow README RoboCasa setup (Linux only)."
        ) from exc
    return gym


def _resolve_gym_env_id(task: str) -> str:
    """Map a Longor task name to the RoboCasa gym registration id."""
    if task != OPEN_DRAWER_TASK:
        raise RuntimeError(
            f"RoboCasa backend supports task {OPEN_DRAWER_TASK!r} only, got {task!r}"
        )
    return OPEN_DRAWER_GYM_ENV_ID


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
        self._env = gym.make(env_id)
        obs, _info = self._env.reset()
        self._last_obs = obs
        instruction = getattr(self._env, "language", "") or getattr(
            self._env.unwrapped, "language", ""
        )
        if not instruction:
            raise RuntimeError(
                "RoboCasa episode language instruction is missing; "
                "refusing to substitute a default task string"
            )
        self.task_description = str(instruction)
        self._steps = 0
        self._writer = None
        logger.info("RoboCasa ready: %r - %r", task, self.task_description)

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self) -> tuple[tuple[bytes, bytes, bytes], list[float]]:
        """Return policy-ordered JPEG views and flattened PandaOmron state."""
        obs = self._last_obs
        if not isinstance(obs, dict):
            raise RuntimeError("RoboCasa observation must be a flat modality dict")
        images: list[bytes] = []
        for cam_key in CAMERA_KEYS:
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
            images.append(to_jpeg(np.ascontiguousarray(arr, dtype=np.uint8)))
        state_groups = self._state_groups_from_obs(obs)
        state = flatten_state(state_groups)
        validate_finite_vector(state, "state")
        return (images[0], images[1], images[2]), state

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
        return bool(terminated or truncated)

    def success(self) -> bool:
        """Return whether the OpenDrawer episode succeeded (sim task oracle)."""
        return self._drawer_open_oracle()

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Evaluate structured subtask completion (OpenDrawer sim oracle)."""
        skill = step["skill"]
        if skill == "open_drawer":
            return self._drawer_open_oracle()
        raise KeyError(f"unsupported skill: {skill!r}")

    def _drawer_open_oracle(self) -> bool:
        """Sim-GT: drawer joint displacement exceeds open threshold."""
        unwrapped = self._env.unwrapped
        sim = getattr(unwrapped, "sim", None)
        if sim is None:
            raise RuntimeError("RoboCasa drawer oracle requires sim access")
        qpos = np.asarray(sim.data.qpos, dtype=np.float64)
        qpos_addr = getattr(unwrapped, "drawer_qpos_addrs", None)
        if qpos_addr is None:
            raise RuntimeError(
                "RoboCasa OpenDrawer drawer oracle requires drawer_qpos_addrs on env"
            )
        drawer_q = float(qpos[int(qpos_addr)])
        threshold = getattr(unwrapped, "drawer_open_threshold", None)
        if threshold is None:
            raise RuntimeError(
                "RoboCasa OpenDrawer drawer oracle requires drawer_open_threshold on env"
            )
        return drawer_q >= float(threshold)

    def transfer_distances(self, step: dict[str, str]) -> tuple[float, float]:
        """Not supported outside VLABench transfer skills."""
        raise KeyError(f"unsupported skill: {step['skill']!r}")

    def close(self) -> None:
        self._env.close()
