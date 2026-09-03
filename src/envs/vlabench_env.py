"""VLABench simulation backend (default sim).

Observation and action layout match VLABench primitive policies: 7-D state
``[pos_robot(3), euler_xyz(3), gripper]``, plus policy views derived from the
simulator camera stack. 7-D absolute EEF actions are converted to native joint
controls in this adapter (not in the executor).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)

_RENDER_RESOLUTION = (480, 480)
_FRANKA_FINGER_OPEN = 0.04
_GRIPPER_OPEN = 0.1  # policy gripper: >= this is open, else closed
_POLICY_CAMERA_SOURCE_COUNT = 4


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


def to_jpeg(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 frame as JPEG bytes."""
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


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
        norm = name.strip().lower().replace(" ", "_")
        if not norm:
            raise ValueError(f"empty object name: {name!r}")
        if name in entities:
            return entities[name]
        if norm in entities:
            return entities[norm]
        raise ValueError(f"unknown object name {name!r}")

    def is_grasping(self, object_name: str) -> bool:
        """True when VLABench reports contact and the gripper is closed."""
        entity = self._resolve_entity(object_name)
        if not entity.is_grasped(self._env.physics, self._env.task.robot):
            return False
        return self.state()[-1] < _GRIPPER_OPEN

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Evaluate structured subtask completion for one executor step."""
        skill = step["skill"]
        if skill == "grab":
            return self.is_grasping(step["object_name"])
        if skill == "place":
            position = step["position"].lower()
            obj = self._resolve_entity(step["object_name"])
            target = self._resolve_entity(step["target_name"])
            obj_pos = np.asarray(obj.get_xpos(self._env.physics), dtype=np.float64)
            try:
                contain_fn = target.contain
            except AttributeError:
                raise ValueError(
                    f"target {step['target_name']!r} is not a receptacle"
                ) from None
            if position in {
                "inside",
                "on",
                "center",
                "left",
                "right",
                "top",
                "bottom",
            }:
                return bool(contain_fn(obj_pos, self._env.physics))
            raise ValueError(f"unsupported place position: {position!r}")
        if skill in {"close", "turn_on"}:
            raise AssertionError(f"{skill!r} not supported for this env")
        raise KeyError(f"unsupported skill: {skill!r}")

    def eef_goal_distance(self, step: dict[str, str]) -> float:
        """Return ||EEF xyz - current skill goal|| from sim GT (debug)."""
        skill = step["skill"]
        physics = self._env.physics
        if skill == "grab":
            goal = np.asarray(
                self._resolve_entity(step["object_name"]).get_xpos(physics),
                dtype=np.float64,
            )
        elif skill == "place":
            goal = np.asarray(
                self._resolve_entity(step["target_name"]).get_xpos(physics),
                dtype=np.float64,
            )
        elif skill in {"close", "turn_on"}:
            raise AssertionError(f"{skill!r} not supported for this env")
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
