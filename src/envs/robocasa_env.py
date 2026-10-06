"""RoboCasa backend for OpenDrawer and KettleBoiling (linux-only lazy import).

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
KETTLE_BOILING_TASK = "kettle_boiling"
KETTLE_BOILING_GYM_ENV_ID = (
    "robocasa_panda_omron/KettleBoiling_PandaOmron_Env"
)
_TASK_GYM_ENV_IDS = {
    OPEN_DRAWER_TASK: OPEN_DRAWER_GYM_ENV_ID,
    KETTLE_BOILING_TASK: KETTLE_BOILING_GYM_ENV_ID,
}
_TASK_HORIZONS = {
    OPEN_DRAWER_TASK: 720,
    KETTLE_BOILING_TASK: 1500,
}
_KETTLE_BURNER_XY_RADIUS = 0.15
_KNOB_ON_ABS_MIN = 0.35

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
            raise ValueError(
                f"state group {key!r} expected {width} floats, got {len(chunk)}")
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
            raise ValueError(
                f"action group {key!r} expected {width} floats, got {len(chunk)}")
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


def burner_phrase_to_key(phrase: str, valid_keys: tuple[str, ...]) -> str:
    """Map a natural-language burner phrase to a stove burner dict key."""
    text = phrase.strip().lower()
    if text.endswith(" of the stove"):
        text = text[: -len(" of the stove")].strip()
    if text.endswith(" burner"):
        text = text[: -len(" burner")].strip()
    key = text.replace(" ", "_")
    if key not in valid_keys:
        listed = ", ".join(sorted(valid_keys))
        raise KeyError(
            f"unknown burner {phrase!r}; valid keys: {listed}")
    return key


def _resolve_gym_env_id(task: str) -> str:
    """Map a Longor task name to the RoboCasa gym registration id."""
    env_id = _TASK_GYM_ENV_IDS.get(task)
    if env_id is None:
        supported = ", ".join(sorted(_TASK_GYM_ENV_IDS))
        raise RuntimeError(
            f"RoboCasa backend supports tasks {supported!r} only, "
            f"got {task!r}"
        )
    return env_id


_TASK_DESCRIPTION_KEY = "annotation.human.action.task_description"


def _task_description_from_reset_obs(reset_obs: object) -> str:
    """Read episode language from the GrootRoboCasaEnv reset observation."""
    if not isinstance(reset_obs, dict):
        raise RuntimeError(
            "RoboCasa reset observation must be a dict with "
            f"{_TASK_DESCRIPTION_KEY!r}"
        )
    raw = reset_obs.get(_TASK_DESCRIPTION_KEY)
    if raw is None:
        raise RuntimeError(
            f"RoboCasa reset observation missing {_TASK_DESCRIPTION_KEY!r}"
        )
    value = raw
    if isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError(
                f"RoboCasa {_TASK_DESCRIPTION_KEY!r} is empty"
            )
        value = value[0]
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            raise RuntimeError(
                f"RoboCasa {_TASK_DESCRIPTION_KEY!r} is empty"
            )
        value = value.flat[0]
    text = str(value).strip()
    if not text:
        raise RuntimeError(
            f"RoboCasa {_TASK_DESCRIPTION_KEY!r} is blank after normalization"
        )
    return text


class RoboCasaEnv:
    """One RoboCasa episode (OpenDrawer or KettleBoiling)."""

    @classmethod
    def from_task(cls, task: str, task_idx: int = 0) -> "RoboCasaEnv":
        """Build one RoboCasa episode.

        ``task_idx`` is reserved for future eval sets.
        """
        if task_idx != 0:
            raise RuntimeError(
                "RoboCasa backend supports task_idx=0 only"
            )
        env = cls.__new__(cls)
        env._init_env(task)
        return env

    def _init_env(self, task: str) -> None:
        env_id = _resolve_gym_env_id(task)
        self._task = task
        gym = _import_robocasa_gym()
        self._env = gym.make(env_id, enable_render=True)
        obs, _info = self._env.reset()
        self._last_obs = obs
        self.task_description = _task_description_from_reset_obs(obs)
        self._steps = 0
        self._writer = None
        self._record()
        logger.info("RoboCasa ready: %r - %r", task, self.task_description)

    @property
    def steps(self) -> int:
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

    def _state_groups_from_obs(
        self, obs: dict[str, object]) -> dict[str, list[float]]:
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
        """Return whether the episode succeeded (sim task oracle)."""
        if self._task == OPEN_DRAWER_TASK:
            return self._drawer_open_oracle()
        if self._task == KETTLE_BOILING_TASK:
            return bool(self._env.unwrapped.env._check_success())
        raise RuntimeError(f"unsupported task {self._task!r}")

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Evaluate structured subtask completion (sim oracle)."""
        skill = step["skill"]
        if self._task == OPEN_DRAWER_TASK:
            if skill == "open":
                return self._drawer_open_oracle()
            raise KeyError(f"unsupported skill: {skill!r}")
        if self._task == KETTLE_BOILING_TASK:
            if skill == "place":
                return self._kettle_place_oracle(step)
            if skill == "turn_on":
                return self._kettle_turn_on_oracle(step)
            raise KeyError(f"unsupported skill: {skill!r}")
        raise RuntimeError(f"unsupported task {self._task!r}")

    def _sim_env(self):
        """Return the inner robosuite env (sim ground truth)."""
        return self._env.unwrapped.env

    def _burner_keys(self, sim_env) -> tuple[str, ...]:
        return tuple(sorted(sim_env.stove.burner_sites.keys()))

    def _resolve_burner_key(self, phrase: str, sim_env) -> str:
        return burner_phrase_to_key(phrase, self._burner_keys(sim_env))

    def _kettle_place_oracle(self, step: dict[str, str]) -> bool:
        """Sim-GT: kettle on named burner, gripper clear (mirrors upstream check)."""
        from robocasa.utils import object_utils as OU

        sim_env = self._sim_env()
        dest = step["destination_name"]
        try:
            burner_key = self._resolve_burner_key(dest, sim_env)
        except KeyError as exc:
            raise ValueError(
                f"place destination_name must be a stove burner phrase: {exc}"
            ) from exc
        stove = sim_env.stove
        if not OU.check_obj_fixture_contact(sim_env, "obj", stove):
            return False
        kettle = sim_env.objects["obj"]
        body_id = sim_env.obj_body_id[kettle.name]
        kettle_xy = sim_env.sim.data.body_xpos[body_id][0:2]
        site_info = stove.burner_sites.get(burner_key)
        if site_info is None:
            return False
        site_name = site_info.get("name")
        if site_name is None:
            return False
        burner_xy = sim_env.sim.data.get_site_xpos(site_name)[0:2]
        if np.linalg.norm(kettle_xy - burner_xy) > _KETTLE_BURNER_XY_RADIUS:
            return False
        return bool(OU.gripper_obj_far(sim_env))

    def _kettle_turn_on_oracle(self, step: dict[str, str]) -> bool:
        """Sim-GT: named burner knob rotated into the on range."""
        sim_env = self._sim_env()
        name = step["target_name"]
        try:
            burner_key = self._resolve_burner_key(name, sim_env)
        except KeyError as exc:
            raise ValueError(
                f"turn_on target_name must be a stove burner phrase: {exc}"
            ) from exc
        knobs_state = sim_env.stove.get_knobs_state(env=sim_env)
        angle = knobs_state[burner_key]
        lo = _KNOB_ON_ABS_MIN
        hi = 2 * math.pi - _KNOB_ON_ABS_MIN
        return lo <= abs(angle) <= hi

    def _drawer_open_oracle(self) -> bool:
        """Sim-GT: robosuite OpenDrawer ``_check_success`` (drawer >= 95% open)."""
        return bool(self._env.unwrapped.env._check_success())

    def close(self) -> None:
        """Release the video writer and close the simulation environment."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("video saved: %s", config.VIDEO_PATH)
        self._env.close()
