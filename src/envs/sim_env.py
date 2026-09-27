"""Sim environment factory for the configured backend."""
from __future__ import annotations

from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from envs.vlabench_env import VLABenchEnv


class SimEnv:
    """Factory that returns a sim backend instance for `config.SIM_ENV`."""

    def __new__(cls, task: str, task_idx: int = 0) -> VLABenchEnv:
        """Create the configured sim backend for the given task."""
        sim = config.SIM_ENV
        if sim == "vlabench":
            from envs.vlabench_env import VLABenchEnv

            return VLABenchEnv.from_evaluation_episode_index(task, task_idx)
        if sim == "robocasa":
            from envs.robocasa_env import RoboCasaEnv

            return RoboCasaEnv.from_task(task, task_idx)
        raise AssertionError(f"{sim!r} not supported for this env")
