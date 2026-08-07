"""LIBERO simulation, wrapping ``OffScreenRenderEnv`` directly.

Observation and action conventions follow the reference OpenPI LIBERO evaluation
loop, because that is what the policies were trained and benchmarked against:
7-D OSC action ``[dx, dy, dz, drx, dry, drz, gripper]``, both camera frames
rotated 180 degrees, and an 8-D proprioceptive state.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


def _import_libero():
    try:
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

    def __init__(self, task_suite: str, task_idx: int) -> None:
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

        self.task_description = str(task.language)
        self.max_steps = config.MAX_EPISODE_STEPS[task_suite]
        self.done = False
        self._steps = 0
        self._writer = None

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
        self._obs = self._env.set_init_state(suite.get_task_init_states(task_idx)[0])
        for _ in range(config.SETTLE_STEPS):
            self.step(config.SETTLE_ACTION)

        logger.info(
            "LIBERO ready: %s task %d - %r", task_suite, task_idx, self.task_description
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
