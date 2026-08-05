"""Wire and run the planner-executor loop for a single LIBERO task."""
from __future__ import annotations

import logging
from typing import Literal

from executor import Executor, StopReason
from libero_env import LiberoEnv
from plan import get_stop_reason, is_a_stop_plan
from planner import LLMPlanner
from skills import SkillRegistry, arm_skill_catalog, register_libero_skills

logger = logging.getLogger(__name__)

RunStatus = Literal[
    "done", "plan_failed", "stopped", "step_budget", "execution_error"
]


def _stringify_optional(value: object | None) -> str:
    """Convert optional values to logging-safe strings."""
    return "" if value is None else str(value)


def _return_indicates_failure(value: str) -> bool:
    """Detect explicit failure phrasing in a plan return value."""
    return "failed to" in value.lower()


class Runner:
    """Orchestrates plan → execute → plan feedback cycles until done or stopped."""

    def __init__(self, goal: str, task_suite: str, task_idx: int) -> None:
        """Wire env, skills, planner, and executor for one LIBERO task."""
        self._goal = goal
        self._env = LiberoEnv(task_suite, task_idx)
        try:
            self._skills = SkillRegistry()
            register_libero_skills(self._skills, self._env)
            scene_objects = self._skills.get("state.list_objects")()
            logger.info("Scene objects: %s", scene_objects)
            skills_catalog = self._skills.render_catalog() + "\n" + arm_skill_catalog()
            self._planner = LLMPlanner(goal, skills_catalog, scene_objects)
            self._executor = Executor(self._skills, env=self._env)
        except Exception:
            self._env.close()
            raise

    def run(self) -> None:
        """Run the task loop and log its outcome."""
        status: RunStatus = "done"
        error = ""
        value = ""
        total_steps = 0
        try:
            status, error, value, total_steps = self._execute_plan_loop()
        except RuntimeError as exc:
            status = "plan_failed"
            error = str(exc)
        finally:
            success = self._verdict(status, value)
            self._log_status(status, success, value, error, total_steps)
            self._env.close()

    def _execute_plan_loop(self) -> tuple[RunStatus, str, str, int]:
        """Iterate plan/execute cycles until done, stop, budget, or execution error."""
        total_steps = 0
        execution_sum = None
        snapshot = self._env.observe()[0]
        while True:
            current_plan = self._planner.plan(snapshot, execution_sum)
            if is_a_stop_plan(current_plan):
                return "stopped", get_stop_reason(current_plan), "", total_steps

            execution_sum = self._executor.run(current_plan)
            total_steps += execution_sum.steps

            if execution_sum.reason == StopReason.PLAN_DONE:
                return "done", "", _stringify_optional(execution_sum.return_value), total_steps

            if execution_sum.reason == StopReason.STEP_BUDGET:
                return "step_budget", "", "", total_steps

            if execution_sum.reason == StopReason.EXECUTION_ERROR:
                return "execution_error", execution_sum.error or "", "", total_steps

            if execution_sum.reason != StopReason.FEEDBACK_POINT:
                msg = f"unexpected stop reason: {execution_sum.reason.value}"
                return "execution_error", msg, "", total_steps

            if (
                execution_sum.feedback_description is None
                or execution_sum.feedback_prompt is None
            ):
                return (
                    "execution_error",
                    "feedback execution summary missing description or prompt",
                    "",
                    total_steps,
                )

            snapshot = self._env.observe()[0]

    def _verdict(self, status: RunStatus, value: str) -> bool:
        """Compute final success verdict from status, return value, and env goal check."""
        if status != "done":
            return False
        if _return_indicates_failure(value):
            return False
        return bool(self._env.success())

    def _log_status(
        self,
        status: RunStatus,
        success: bool,
        value: str,
        error: str,
        total_steps: int,
    ) -> None:
        """Emit outcome-specific logs and a compact run summary."""
        if success:
            logger.info("Task succeeded: %r", self._goal)
        elif status == "done":
            if _return_indicates_failure(value):
                logger.info(
                    "Task not achieved: %r (plan finished; skill reported failure)",
                    self._goal,
                )
            else:
                logger.info(
                    "Task not achieved: %r (plan finished; env goal check failed)",
                    self._goal,
                )
        elif status == "plan_failed":
            logger.info("Task failed: %r (planner error during replan)", self._goal)
        elif status == "stopped":
            logger.info("Task stopped: %r (%s)", self._goal, error)
        elif status == "step_budget":
            logger.info("Task failed: %r (executor step budget exceeded)", self._goal)
        else:
            logger.info("Task failed: %r (status=%s)", self._goal, status)

        summary_parts = [
            f"success={success}",
            f"plan_status={status}",
            f"steps={total_steps}",
        ]
        if value:
            summary_parts.append(f"last_skill={value!r}")
        if error:
            summary_parts.append(f"error={error}")

        logger.info("Run summary: %s", ", ".join(summary_parts))
