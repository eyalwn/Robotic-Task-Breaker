"""Wire and run single-pass planner-executor for one LIBERO task."""
from __future__ import annotations

import logging
from typing import Literal

from executor import Executor, StopReason
from libero_env import LiberoEnv
from plan import get_stop_reason, is_stop_plan
from planner import LLMPlanner
from skills import skill_catalog

logger = logging.getLogger(__name__)

RunStatus = Literal["done", "plan_failed", "stopped", "execution_error"]


class Runner:
    """Orchestrates a single observe → plan → execute pass for one LIBERO task."""

    def __init__(self, task_suite: str, task_idx: int) -> None:
        """Wire env, planner, and executor for one LIBERO task."""
        self._env = LiberoEnv(task_suite, task_idx)
        self._goal = self._env.task_description
        try:
            self._planner = LLMPlanner(self._goal, skill_catalog())
            self._executor = Executor(env=self._env)
        except Exception:
            self._env.close()
            raise

    def run(self) -> None:
        """Run the task loop and log its outcome."""
        status: RunStatus = "plan_failed"
        error = ""
        total_steps = 0
        try:
            status, error, total_steps = self._execute_once()
        except RuntimeError as exc:
            error = str(exc)
        finally:
            success = self._verdict(status)
            self._log_status(status, success, error, total_steps)
            self._env.close()

    def _execute_once(self) -> tuple[RunStatus, str, int]:
        """Observe once, plan once, and execute the plan."""
        snapshot = self._env.observe()[0]
        current_plan = self._planner.plan(snapshot)
        if is_stop_plan(current_plan):
            return "stopped", get_stop_reason(current_plan), 0

        execution_sum = self._executor.run(current_plan)
        total_steps = execution_sum.steps

        if execution_sum.reason == StopReason.PLAN_DONE:
            return "done", "", total_steps

        if execution_sum.reason == StopReason.SUBTASK_TIMEOUT:
            return "execution_error", "subtask step cap exceeded", total_steps

        if execution_sum.reason == StopReason.EXECUTION_ERROR:
            return "execution_error", execution_sum.error or "", total_steps

        msg = f"unexpected stop reason: {execution_sum.reason}"
        return "execution_error", msg, total_steps

    def _verdict(self, status: RunStatus) -> bool:
        """Compute final success verdict from status and env goal check."""
        if status != "done":
            return False
        return bool(self._env.success())

    def _log_status(
        self,
        status: RunStatus,
        success: bool,
        error: str,
        total_steps: int,
    ) -> None:
        """Emit outcome-specific logs and a compact run summary."""
        if success:
            logger.info("Task succeeded: %r", self._goal)
        elif status == "done":
            logger.info(
                "Task not achieved: %r (plan finished; env goal check failed)",
                self._goal,
            )
        elif status == "plan_failed":
            logger.info("Task failed: %r (planner error)", self._goal)
        elif status == "stopped":
            logger.info("Task stopped: %r (%s)", self._goal, error)
        else:
            logger.info("Task failed: %r (status=%s)", self._goal, status)

        summary_parts = [
            f"success={success}",
            f"plan_status={status}",
            f"steps={total_steps}",
        ]
        if error:
            summary_parts.append(f"error={error}")

        logger.info("Run summary: %s", ", ".join(summary_parts))
