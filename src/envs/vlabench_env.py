"""VLABench simulation backend (default sim).

Observation and action layout match VLABench primitive policies: 7-D state
``[pos_robot(3), euler_xyz(3), gripper]``, plus policy views derived from the
simulator camera stack. 7-D absolute EEF actions are converted to native joint
controls in this adapter (not in the executor).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)

_RENDER_RESOLUTION = (480, 480)
_FRANKA_FINGER_OPEN = 0.04
_GRIPPER_OPEN = 0.1  # policy gripper: >= this is open, else closed
_POLICY_CAMERA_SOURCE_COUNT = 4
_TRACK_RELATIVE_PATH = (
    Path("configs") / "evaluation" / "tracks" / "track_1_in_distribution.json"
)
_COOK_DISHES_EVAL_FILE = Path(__file__).with_name(
    "cook_dishes_evaluation_episodes.json"
)
_COOK_DISHES_TASK = "cook_dishes"


def _import_vlabench():
    try:
        import VLABench.robots  # noqa: F401
        import VLABench.tasks  # noqa: F401
        from VLABench.envs import load_env
    except ImportError as exc:
        raise ImportError(
            "VLABench is not installed. Follow README §2 (App venv / VLAbench)."
        ) from exc
    return load_env


def _import_vlabench_package():
    try:
        import VLABench
    except ImportError as exc:
        raise ImportError(
            "VLABench is not installed. Follow README §2 (App venv / VLAbench)."
        ) from exc
    return VLABench


def _resolve_track1_path() -> Path:
    """Resolve Track-1 config path from the installed VLABench package root."""
    vlabench_pkg = _import_vlabench_package()
    module_file = getattr(vlabench_pkg, "__file__", "")
    if not module_file:
        raise RuntimeError("Unable to resolve installed VLABench package location")
    return Path(module_file).resolve().parent / _TRACK_RELATIVE_PATH


def get_track1_episode_configs(task: str) -> list[dict[str, object]]:
    """Return all deterministic Track-1 configs for one task."""
    track_path = _resolve_track1_path()
    if not track_path.is_file():
        raise RuntimeError(f"Track-1 file not found: {track_path}")
    with track_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    configs = payload.get(task)
    if not isinstance(configs, list) or not configs:
        raise RuntimeError(f"Track-1 file has no episodes for task {task!r}")
    return [dict(config) for config in configs]


def get_track1_episode_config(task: str, task_idx: int) -> dict[str, object]:
    """Return one deterministic Track-1 episode config by zero-based index."""
    configs = get_track1_episode_configs(task)
    if task_idx < 0 or task_idx >= len(configs):
        raise RuntimeError(
            "Invalid Track-1 task_idx="
            f"{task_idx} for task={task!r}; available_config_count={len(configs)}"
        )
    return dict(configs[task_idx])


def _canonical_episode_json(episode: dict[str, object]) -> str:
    """Return one episode config as canonical JSON (sorted keys, compact)."""
    return json.dumps(episode, sort_keys=True, separators=(",", ":"))


def _episode_sha256(episode: dict[str, object]) -> str:
    """Return the SHA-256 digest of one canonical episode config payload."""
    return hashlib.sha256(_canonical_episode_json(episode).encode("utf-8")).hexdigest()


def _load_cook_dishes_evaluation_episodes() -> list[dict[str, object]]:
    """Return the frozen official cook_dishes evaluation episode configs."""
    if not _COOK_DISHES_EVAL_FILE.is_file():
        raise RuntimeError(
            f"cook_dishes evaluation file not found: {_COOK_DISHES_EVAL_FILE}"
        )
    with _COOK_DISHES_EVAL_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(
            f"cook_dishes evaluation file has no episodes: {_COOK_DISHES_EVAL_FILE}"
        )
    provenance = payload.get("provenance")
    expected_hashes = (
        provenance.get("episode_sha256") if isinstance(provenance, dict) else None
    )
    if not isinstance(expected_hashes, list) or len(expected_hashes) != len(episodes):
        raise RuntimeError(
            "cook_dishes evaluation provenance missing episode_sha256 list"
        )
    episodes_out = [dict(episode) for episode in episodes]
    actual_hashes = [_episode_sha256(episode) for episode in episodes_out]
    if actual_hashes != expected_hashes:
        raise RuntimeError("cook_dishes evaluation episode integrity check failed")
    return episodes_out


def get_evaluation_episode_configs(task: str) -> list[dict[str, object]]:
    """Return evaluation episode configs for one task (Track-1 or fixed M3 set)."""
    if task == _COOK_DISHES_TASK:
        return _load_cook_dishes_evaluation_episodes()
    return get_track1_episode_configs(task)


def get_evaluation_episode_config(task: str, task_idx: int) -> dict[str, object]:
    """Return one evaluation episode config by zero-based index."""
    configs = get_evaluation_episode_configs(task)
    if task_idx < 0 or task_idx >= len(configs):
        raise RuntimeError(
            "Invalid evaluation task_idx="
            f"{task_idx} for task={task!r}; available_config_count={len(configs)}"
        )
    return dict(configs[task_idx])


def validate_evaluation_episode_range(
    task: str,
    task_idx: int,
    episodes: int,
) -> None:
    """Fail fast when a multi-episode campaign index range is invalid."""
    if task_idx < 0:
        raise RuntimeError(
            f"Invalid evaluation task_idx={task_idx} for task={task!r}; must be >= 0"
        )
    available = len(get_evaluation_episode_configs(task))
    last_idx = task_idx + episodes - 1
    if last_idx >= available:
        raise RuntimeError(
            "Invalid evaluation range "
            f"task_idx={task_idx} episodes={episodes} for task={task!r}; "
            f"available_config_count={available}"
        )


def to_jpeg(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 frame as JPEG bytes."""
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _normalize_entity_name(name: str) -> str:
    """Normalize one planner or sim entity name for lookup."""
    text = name.strip().lower().replace("_", " ")
    if "seen" in text:
        text = text.replace("seen", "")
    return "_".join(text.split())


def _build_entity_aliases(entity_names: dict[str, object]) -> dict[str, str]:
    """Map normalized NL names to sim entity keys for oracle resolution."""
    aliases: dict[str, str] = {}
    for sim_name in entity_names:
        nl_key = _normalize_entity_name(sim_name)
        if nl_key and nl_key not in entity_names:
            aliases[nl_key] = sim_name
    return aliases


class VLABenchEnv:
    """One VLABench task episode: build, observe, step, check success."""

    def __init__(self, task: str) -> None:
        self._init_env(task, episode_config={})

    @classmethod
    def from_track_episode(
        cls,
        task: str,
        episode_config: dict[str, object],
    ) -> "VLABenchEnv":
        """Build one deterministic evaluation episode from a track config entry."""
        env = cls.__new__(cls)
        env._init_env(task, episode_config=episode_config)
        return env

    @classmethod
    def from_track_episode_index(cls, task: str, task_idx: int) -> "VLABenchEnv":
        """Build one deterministic Track-1 held-out episode by task/index."""
        episode_config = get_track1_episode_config(task, task_idx)
        return cls.from_track_episode(task, episode_config)

    @classmethod
    def from_evaluation_episode_index(cls, task: str, task_idx: int) -> "VLABenchEnv":
        """Build one evaluation episode by task/index (Track-1 or fixed M3 set)."""
        episode_config = get_evaluation_episode_config(task, task_idx)
        return cls.from_track_episode(task, episode_config)

    def _init_env(
        self,
        task: str,
        episode_config: dict[str, object],
    ) -> None:
        """Initialize the wrapped VLABench environment."""
        load_env = _import_vlabench()
        h, w = _RENDER_RESOLUTION
        kwargs: dict[str, object] = {
            "task": task,
            "render_resolution": (h, w),
        }
        if episode_config:
            kwargs["episode_config"] = episode_config
            kwargs["random_init"] = False
            kwargs["run_mode"] = "eval"
        self._env = load_env(**kwargs)
        instruction = self._env.task.get_instruction()
        if not instruction:
            raise RuntimeError("VLABench task has no language instruction")
        self.task_description = str(instruction)
        self._robot_base_xyz = np.asarray(
            self._env.get_robot_frame_position(), dtype=np.float64
        ).reshape(3)
        self._steps = 0
        self._writer = None
        entities = getattr(self._env.task, "entities", None) or {}
        self._entity_aliases = _build_entity_aliases(entities)
        logger.info("VLABench ready: %r - %r", task, self.task_description)

    @property
    def steps(self) -> int:
        return self._steps

    def _state_from_obs(self, obs) -> list[float]:
        """Build 7-D proprioceptive state from one VLABench observation dict."""
        from scipy.spatial.transform import Rotation

        if "ee_state" not in obs:
            raise RuntimeError("VLABench observation missing 'ee_state'")
        raw = np.asarray(obs["ee_state"], dtype=np.float64).ravel()
        if raw.size < 8:
            raise RuntimeError(
                f"VLABench ee_state too short: expected 8, got {raw.size}"
            )
        pos_world = raw[:3]
        quat_wxyz = raw[3:7]
        gripper = float(raw[7])
        pos_robot = pos_world - self._robot_base_xyz
        euler_xyz = Rotation.from_quat(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        ).as_euler("xyz", degrees=False)
        state = np.concatenate([pos_robot, euler_xyz, [gripper]]).astype(np.float64)
        return [float(v) for v in state]

    def state(self) -> list[float]:
        """7-D proprioceptive state: robot-frame xyz, euler xyz, gripper."""
        obs = self._env.get_observation(require_pcd=False)
        return self._state_from_obs(obs)

    def observe(self) -> tuple[tuple[bytes, bytes, bytes], list[float]]:
        """Return policy-ordered JPEG views plus 7-D proprioceptive state."""
        obs = self._env.get_observation(require_pcd=False)
        rgb = obs.get("rgb")
        if rgb is None:
            raise RuntimeError("VLABench observation missing 'rgb' cameras")
        arr = np.asarray(rgb)
        if arr.ndim != 4:
            raise RuntimeError(f"unexpected rgb shape: {arr.shape}")
        if arr.shape[0] < _POLICY_CAMERA_SOURCE_COUNT:
            raise RuntimeError(
                f"expected {_POLICY_CAMERA_SOURCE_COUNT} cameras, got {arr.shape[0]}"
            )
        # OpenPI's VLABench evaluator maps second/front/wrist from rgb[0]/[2]/[3].
        front = np.ascontiguousarray(arr[2])
        second = np.ascontiguousarray(arr[0])
        wrist = np.ascontiguousarray(arr[3])
        return (
            (to_jpeg(front), to_jpeg(second), to_jpeg(wrist)),
            self._state_from_obs(obs),
        )

    def step(self, action) -> bool:
        """Apply one 7-D EEF action and return whether the task is now solved."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"expected 7-D EEF action, got shape {action.shape}")

        from dm_control.utils.inverse_kinematics import qpos_from_site_pose
        from scipy.spatial.transform import Rotation

        physics = self._env.physics
        ctrl_dim = int(physics.data.ctrl.shape[0])
        pos_world = action[:3] + self._robot_base_xyz
        rx, ry, rz = float(action[3]), float(action[4]), float(action[5])
        gripper = float(action[6])
        qxyzw = Rotation.from_euler("xyz", [rx, ry, rz], degrees=False).as_quat()
        quat = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float64)

        robot = self._env.task.robot
        site_name = robot.end_effector_site.full_identifier
        ik_result = qpos_from_site_pose(
            physics,
            site_name=site_name,
            target_pos=pos_world,
            target_quat=quat,
            inplace=False,
            max_steps=100,
        )
        n_dof = robot.n_dof
        ctrl = np.zeros(ctrl_dim, dtype=np.float64)
        ctrl[:n_dof] = ik_result.qpos[:n_dof]
        finger_open = _FRANKA_FINGER_OPEN if gripper >= _GRIPPER_OPEN else 0.0
        ctrl[n_dof:] = finger_open

        self._env.step(ctrl)
        self._steps += 1
        self._record()
        return self.success()

    def success(self) -> bool:
        return bool(self._env.task.should_terminate_episode(self._env.physics))

    def _resolve_entity(self, name: str):
        entities = self._env.task.entities
        norm = _normalize_entity_name(name)
        if not norm:
            raise ValueError(f"empty object name: {name!r}")
        if name in entities:
            return entities[name]
        if norm in entities:
            return entities[norm]
        sim_name = self._entity_aliases.get(norm)
        if sim_name is not None:
            return entities[sim_name]
        raise ValueError(f"unknown object name {name!r}")

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Evaluate structured subtask completion for one executor step."""
        skill = step["skill"]
        if skill == "transfer":
            obj = self._resolve_entity(step["object_name"])
            target = self._resolve_entity(step["target_name"])
            obj_pos = np.asarray(obj.get_xpos(self._env.physics), dtype=np.float64)
            try:
                contain_fn = target.contain
            except AttributeError:
                raise ValueError(
                    f"target {step['target_name']!r} is not a receptacle"
                ) from None
            return bool(contain_fn(obj_pos, self._env.physics))
        raise KeyError(f"unsupported skill: {skill!r}")

    def eef_goal_distance(self, step: dict[str, str]) -> float:
        """Return ||EEF xyz - current skill goal|| from sim GT (debug)."""
        skill = step["skill"]
        physics = self._env.physics
        if skill == "transfer":
            goal = np.asarray(
                self._resolve_entity(step["target_name"]).get_xpos(physics),
                dtype=np.float64,
            )
        else:
            raise KeyError(f"unsupported skill: {skill!r}")
        obs = self._env.get_observation(require_pcd=False)
        if "ee_state" not in obs:
            raise RuntimeError("VLABench observation missing 'ee_state'")
        raw = np.asarray(obs["ee_state"], dtype=np.float64).ravel()
        if raw.size < 3:
            raise RuntimeError(
                f"VLABench ee_state too short for EEF xyz: got {raw.size}"
            )
        eef = raw[:3]
        return float(np.linalg.norm(eef - goal))

    def _record(self) -> None:
        if not config.SAVE_VIDEO:
            return
        import cv2

        obs = self._env.get_observation(require_pcd=False)
        arr = np.asarray(obs["rgb"])
        frame = np.ascontiguousarray(arr[0])
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

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("video saved: %s", config.VIDEO_PATH)
        self._env.close()
