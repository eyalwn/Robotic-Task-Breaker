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

    def close(self) -> None:
        """Release the backend."""
        self._env.close()
