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
            from libero.libero.envs import OffScreenRenderEnv, predicates
    except ImportError as exc:
        raise ImportError(
            "LIBERO is not installed (Linux only). Install with: "
            "pip install -r deps/requirements-libero.txt"
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv, predicates.get_predicate_fn


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
        benchmark, get_libero_path, OffScreenRenderEnv, get_predicate_fn = _import_libero()
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
        self._get_predicate_fn = get_predicate_fn

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

    def _get_sim_gt(self):
        """Return the inner LIBERO problem env (sim ground truth lives here)."""
        return self._env.env

    def goal_conjuncts(self) -> list[list[str]]:
        """Return a copy of the parsed BDDL goal conjuncts for this task."""
        return [list(conjunct) for conjunct in self._get_sim_gt().parsed_problem["goal_state"]]

    def eval_goal_conjunct(self, index: int) -> bool:
        """Evaluate one BDDL goal conjunct by index."""
        conjuncts = self._get_sim_gt().parsed_problem["goal_state"]
        if index < 0 or index >= len(conjuncts):
            raise ValueError(
                f"goal conjunct index {index} out of range (have {len(conjuncts)})"
            )
        return bool(self.eval_predicate(*conjuncts[index]))

    def eval_predicate(self, predicate: str, *args: str) -> bool:
        """Evaluate an arbitrary LIBERO predicate tuple (lowercase name)."""
        predicate_name = predicate.lower()
        if len(args) not in (1, 2):
            raise ValueError(
                f"predicate needs 1 or 2 object args, got {len(args)} for {predicate!r}"
            )
        states = self._get_sim_gt().object_states_dict
        try:
            predicate_fn = self._get_predicate_fn(predicate_name)
            object_state_1 = states[args[0]]
            if len(args) == 1:
                return bool(predicate_fn(object_state_1))
            object_state_2 = states[args[1]]
            return bool(predicate_fn(object_state_1, object_state_2))
        except KeyError as exc:
            raise ValueError(
                f"unknown object for predicate {predicate_name!r}: {exc.args[0]!r}"
            ) from exc

    def is_grasping(self, object_name: str) -> bool:
        """Return whether both fingerpads contact ``object_name`` geoms."""
        env_gt = self._get_sim_gt()
        obj = env_gt.get_object(object_name)
        if obj is None:
            raise ValueError(f"unknown object for grasp check: {object_name!r}")
        object_geoms = obj.contact_geoms
        gripper = env_gt.robots[0].gripper
        important_geoms = getattr(gripper, "important_geoms", {})
        left_fingerpad = list(important_geoms.get("left_fingerpad", []))
        right_fingerpad = list(important_geoms.get("right_fingerpad", []))
        if left_fingerpad and right_fingerpad:
            return bool(
                env_gt.check_contact(left_fingerpad, object_geoms)
                and env_gt.check_contact(right_fingerpad, object_geoms)
            )
        # Fallback for non-standard grippers where fingerpad groups are missing.
        return bool(env_gt.check_contact(gripper.contact_geoms, object_geoms))

    def eef_goal_distance(self, step: dict[str, str]) -> float:
        """Return ||EEF xyz - current skill goal site|| from sim GT (debug)."""
        skill = step["skill"]
        if skill == "grab":
            name = self._resolve_name(step["object_name"])
        elif skill == "place":
            name = self._resolve_name(
                step["target_name"],
                prefer_region=step["position"].lower() == "inside",
            )
        elif skill in {"close", "turn_on"}:
            name = self._resolve_name(step["target_name"])
        else:
            raise KeyError(f"unsupported skill: {skill!r}")
        goal = np.asarray(
            self._get_sim_gt().object_states_dict[name].get_geom_state()["pos"],
            dtype=np.float64,
        ).flatten()
        eef = np.asarray(self._obs["robot0_eef_pos"], dtype=np.float64).flatten()
        return float(np.linalg.norm(eef - goal))

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Evaluate structured subtask completion for one executor step."""
        skill = step["skill"]
        if skill == "grab":
            return self.is_grasping(self._resolve_name(step["object_name"]))
        if skill == "place":
            return self._place_subtask_satisfied(step)
        if skill == "close":
            return self.eval_predicate("close", self._resolve_name(step["target_name"]))
        if skill == "turn_on":
            return self.eval_predicate("turnon", self._resolve_name(step["target_name"]))
        raise KeyError(f"unsupported skill: {skill!r}")

    def _place_subtask_satisfied(self, step: dict[str, str]) -> bool:
        """Return whether a place step's structured predicate holds."""
        position = step["position"].lower()
        obj = self._resolve_name(step["object_name"])
        prefer_region = position == "inside"
        target = self._resolve_name(step["target_name"], prefer_region=prefer_region)
        if position == "inside":
            return self.eval_predicate("in", obj, target)
        if position in {"on", "center", "left", "right", "top", "bottom"}:
            return self.eval_predicate("on", obj, target)
        raise ValueError(f"unsupported place position: {position!r}")

    def _resolve_name(self, name: str, *, prefer_region: bool = False) -> str:
        """Map an NL or GT name to a unique ``object_states_dict`` key."""
        keys = list(self._get_sim_gt().object_states_dict.keys())
        norm = _normalize_object_name(name)
        if not norm:
            raise ValueError(f"empty object name: {name!r}")
        if name in keys:
            return name
        if norm in keys:
            return norm

        candidates = [key for key in keys if key == norm or key.startswith(norm + "_")]
        if prefer_region:
            region_hits = [key for key in candidates if _is_region_key(key)]
            if region_hits:
                candidates = region_hits
        else:
            candidates = [key for key in candidates if not _is_region_key(key)]

        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            sample = ", ".join(repr(k) for k in keys[:8])
            more = "" if len(keys) <= 8 else f", ... ({len(keys)} total)"
            raise ValueError(
                f"unknown object name {name!r} (normalized {norm!r}); "
                f"available: {sample}{more}"
            )
        raise ValueError(
            f"ambiguous object name {name!r} (normalized {norm!r}); "
            f"candidates: {candidates}"
        )

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


def _normalize_object_name(name: str) -> str:
    """Lowercase and turn spaces into underscores for GT-ID matching."""
    return name.strip().lower().replace(" ", "_")


def _is_region_key(key: str) -> bool:
    """Return whether a GT key looks like a LIBERO region site."""
    return key.endswith("_contain_region") or key.endswith("_place_region")
