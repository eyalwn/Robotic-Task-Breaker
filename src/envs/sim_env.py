"""Sim environment factory: lazy-imports the configured backend."""
from __future__ import annotations

from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from envs.vlabench_env import VLABenchEnv


class SimEnv:
    """Factory that returns a sim backend instance for the configured SIM_ENV."""

    def __new__(cls, task: str, task_idx: int | None) -> VLABenchEnv:
        sim = config.SIM_ENV
        if sim == "vlabench":
            from envs.vlabench_env import VLABenchEnv

            return VLABenchEnv(task)
        if sim == "libero":
            from envs.libero_env import LiberoEnv

            return LiberoEnv(task, task_idx)
        raise AssertionError(f"{sim!r} not supported for this env")
