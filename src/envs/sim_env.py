"""Thin facade over the RoboCasa sim backend."""
from __future__ import annotations

import config


class SimEnv:
    """Forwards the public env API to the backend selected by ``config.SIM_ENV``."""

    def __init__(self, task: str, task_idx: int = 0) -> None:
        """Build that backend and store it on ``self._env``."""
        sim = config.SIM_ENV
        if sim == "robocasa":
            from envs.robocasa_env import RoboCasaEnv

            self._env = RoboCasaEnv.from_task(task, task_idx)
        else:
            raise AssertionError(f"{sim!r} not supported for this env")

    @property
    def steps(self) -> int:
        """Episode step count from the backend."""
        return self._env.steps

    @property
    def horizon(self) -> int:
        """Episode step cap from the backend."""
        return self._env.horizon

    @property
    def task_description(self) -> str:
        """Language instruction for this episode."""
        return self._env.task_description

    def observe(self) -> tuple[tuple[bytes, bytes, bytes], list[float]]:
        """Return ``(images, state)`` from the backend."""
        return self._env.observe()

    def step(self, action: list[float]) -> bool:
        """Forward one action to the backend."""
        return self._env.step(action)

    def success(self) -> bool:
        """Return the backend goal-success flag."""
        return self._env.success()

    def check_subtask(self, step: dict[str, str]) -> bool:
        """Return the backend subtask-completion check for ``step``."""
        return self._env.check_subtask(step)

    def drawer_joint_state(self) -> dict[str, float]:
        """Return drawer-joint state for debug scripts on supported backends."""
        getter = getattr(self._env, "drawer_joint_state", None)
        if getter is None:
            raise RuntimeError("'drawer_joint_state' not supported for this env")
        return getter()

    def episode_wrapper_state(self) -> dict[str, int | bool]:
        """Return wrapper counters and latches for paired-start observability."""
        getter = getattr(self._env, "episode_wrapper_state", None)
        if getter is None:
            raise RuntimeError("'episode_wrapper_state' not supported for this env")
        return getter()

    def capture_episode_checkpoint(self) -> dict[str, object]:
        """Capture a post-reset backend checkpoint for paired arm replay."""
        getter = getattr(self._env, "capture_episode_checkpoint", None)
        if getter is None:
            raise RuntimeError(
                "'capture_episode_checkpoint' not supported for this env")
        return getter()

    def restore_episode_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Restore a backend-owned checkpoint on this env."""
        getter = getattr(self._env, "restore_episode_checkpoint", None)
        if getter is None:
            raise RuntimeError(
                "'restore_episode_checkpoint' not supported for this env")
        getter(checkpoint)

    def close(self) -> None:
        """Release the backend."""
        self._env.close()
