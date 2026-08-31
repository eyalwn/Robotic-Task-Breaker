"""LIBERO simulation, wrapping ``OffScreenRenderEnv`` directly.

Observation and action conventions follow the reference OpenPI LIBERO evaluation
loop, because that is what the policies were trained and benchmarked against:
7-D OSC action ``[dx, dy, dz, drx, dry, drz, gripper]``, both camera frames
rotated 180 degrees, and an 8-D proprioceptive state.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import logging
import math
import os
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


@contextmanager
def _auto_decline_dependency_prompt():
    """Answer ``n`` to dependency ``input()`` prompts to keep runs non-interactive."""
    original_input = builtins.input

    def _decline(prompt: object = "") -> str:
        prompt_text = str(prompt).strip()
        if prompt_text:
            logger.warning("auto-declined dependency prompt: %s", prompt_text)
        return "n"

    builtins.input = _decline
    try:
        yield
    finally:
        builtins.input = original_input


def _import_libero():
    try:
        with _auto_decline_dependency_prompt():
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "LIBERO is not installed (Linux only). Install with: "
            "pip install -r deps/requirements-libero.txt"
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv


def quat2axisangle(quat) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to axis-angle. Ported from robosuite."""
    q = np.asarray(quat, dtype=np.float64).flatten()
    w = min(1.0, max(-1.0, float(q[3])))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(w) / den).astype(np.float32)


def to_jpeg(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 frame as JPEG bytes."""
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


class LiberoEnv:
    """One LIBERO task episode: build, observe, step, check success."""

    def __init__(
        self, task_suite: str, task_idx: int, init_state_idx: int = 0
    ) -> None:
        benchmark, get_libero_path, OffScreenRenderEnv = _import_libero()
        if task_suite not in config.MAX_EPISODE_STEPS:
            raise ValueError(
                f"unknown task_suite {task_suite!r}; "
                f"expected one of {sorted(config.MAX_EPISODE_STEPS)}"
            )
        suite = benchmark.get_benchmark_dict()[task_suite]()
        task = suite.get_task(task_idx)
        bddl_path = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        init_states = suite.get_task_init_states(task_idx)
        count = len(init_states)
        if init_state_idx < 0 or init_state_idx >= count:
            raise ValueError(
                f"init_state_idx {init_state_idx} out of range for "
                f"{task_suite!r} task {task_idx} (have {count} initial states)"
            )

        self.task_description = str(task.language)
        self.max_steps = config.MAX_EPISODE_STEPS[task_suite]
        self.done = False
        self._steps = 0
        self._writer = None
        self._init_states = init_states
        self._init_state_idx = init_state_idx

        with _auto_decline_dependency_prompt():
            self._env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=config.RESOLUTION,
                camera_widths=config.RESOLUTION,
            )
        # Seeding matters even with a fixed initial state: it perturbs object poses.
        self._env.seed(config.SEED)
        self._env.reset()
        # The benchmark's canonical initial states are part of the eval protocol;
        # a bare reset() leaves the scene off the distribution policies trained on.
        self._obs = self._env.set_init_state(init_states[init_state_idx])
        for _ in range(config.SETTLE_STEPS):
            self.step(config.SETTLE_ACTION)

        logger.info(
            "LIBERO ready: %s task %d init %d/%d - %r",
            task_suite,
            task_idx,
            init_state_idx,
            count,
            self.task_description,
        )

    @property
    def init_state_idx(self) -> int:
        """Index of the initial state used for this episode."""
        return self._init_state_idx

    def init_state_count(self) -> int:
        """Return how many canonical initial states exist for this task."""
        return len(self._init_states)

    @staticmethod
    def init_state_idx_for_episode(episode_idx: int, init_state_count: int) -> int:
        """Map a campaign episode index to an initial-state index."""
        if init_state_count <= 0:
            raise ValueError("init_state_count must be positive")
        if episode_idx < 0:
            raise ValueError(f"episode_idx must be >= 0, got {episode_idx}")
        return episode_idx % init_state_count

    def _inner(self):
        """Return the inner LIBERO problem env (sim ground truth lives here)."""
        return self._env.env

    def goal_conjuncts(self) -> list[list[str]]:
        """Return a copy of the parsed BDDL goal conjuncts for this task."""
        return [list(conjunct) for conjunct in self._inner().parsed_problem["goal_state"]]

    def eval_goal_conjunct(self, index: int) -> bool:
        """Evaluate one BDDL goal conjunct by index."""
        conjuncts = self._inner().parsed_problem["goal_state"]
        if index < 0 or index >= len(conjuncts):
            raise ValueError(
                f"goal conjunct index {index} out of range (have {len(conjuncts)})"
            )
        return bool(self._inner()._eval_predicate(conjuncts[index]))

    def eval_predicate(self, predicate: str, *args: str) -> bool:
        """Evaluate an arbitrary LIBERO predicate tuple (lowercase name)."""
        state = [predicate.lower(), *args]
        if len(state) not in (2, 3):
            raise ValueError(
                f"predicate needs 1 or 2 object args, got {len(args)} for {predicate!r}"
            )
        return bool(self._inner()._eval_predicate(state))

    def is_grasping(self, object_name: str) -> bool:
        """Return whether the gripper is grasping ``object_name`` (robosuite check)."""
        inner = self._inner()
        return bool(
            inner._check_grasp(
                gripper=inner.robots[0].gripper,
                object_geoms=inner.get_object(object_name).contact_geoms,
            )
        )

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Evaluate structured subtask completion for one executor step."""
        skill = step["skill"]
        if skill == "grab":
            return self.is_grasping(step["object_name"])
        if skill == "place":
            return self._place_subtask_satisfied(step)
        if skill == "close":
            return self.eval_predicate("close", step["target_name"])
        if skill == "turn_on":
            return self.eval_predicate("turnon", step["target_name"])
        raise KeyError(f"unsupported skill: {skill!r}")

    def _place_subtask_satisfied(self, step: dict[str, str]) -> bool:
        """Return whether a place step's structured predicate holds."""
        position = step["position"].lower()
        obj = step["object_name"]
        target = step["target_name"]
        if position == "inside":
            return self.eval_predicate("in", obj, target)
        if position in {"on", "center", "left", "right", "top", "bottom"}:
            return self.eval_predicate("on", obj, target)
        raise ValueError(f"unsupported place position: {position!r}")

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self) -> tuple[bytes, bytes, list[float]]:
        """Return agentview JPEG, wrist JPEG, and the 8-D proprioceptive state.

        Both frames are rotated 180 degrees: MuJoCo renders bottom-up, and the
        policies' training preprocessing applied the same rotation.
        """
        obs = self._obs
        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        return to_jpeg(image), to_jpeg(wrist), self.state()

    def state(self) -> list[float]:
        """8-D proprioceptive state: eef xyz, eef axis-angle, gripper qpos."""
        obs = self._obs
        values = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).flatten(),
                quat2axisangle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).flatten(),
            ]
        )
        return [float(v) for v in values]

    def step(self, action) -> bool:
        """Apply one 7-D action and return whether the task is now solved."""
        self._obs, _reward, self.done, _info = self._env.step(list(action))
        self._steps += 1
        self._record()
        return self.success()

    def success(self) -> bool:
        return bool(self._env.check_success())

    def _record(self) -> None:
        """Append the policy's view of the current frame to the run video."""
        if not config.SAVE_VIDEO:
            return
        import cv2

        frame = np.ascontiguousarray(self._obs["agentview_image"][::-1, ::-1])
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
