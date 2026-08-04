"""Composition root: wires env, planner, executor, and the run loop."""
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


def _as_str(value: object | None) -> str:
    return "" if value is None else str(value)


def _return_indicates_failure(value: str) -> bool:
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
            status, error, value, total_steps = self._run_loop()
        except RuntimeError as exc:
            status = "plan_failed"
            error = str(exc)
        finally:
            success = self._verdict(status, value)
            self._log_status(status, success, value, error, total_steps)
            self._env.close()

    def _run_loop(self) -> tuple[RunStatus, str, str, int]:
        total_steps = 0
        stop = None
        snapshot = self._env.observe()[0]
        while True:
            current_plan = self._planner.plan(snapshot, stop)
            if is_a_stop_plan(current_plan):
                return "stopped", get_stop_reason(current_plan), "", total_steps

            stop = self._executor.run(current_plan)
            total_steps += stop.steps

            if stop.reason == StopReason.PLAN_DONE:
                return "done", "", _as_str(stop.return_value), total_steps

            if stop.reason == StopReason.STEP_BUDGET:
                return "step_budget", "", "", total_steps

            if stop.reason == StopReason.EXECUTION_ERROR:
                return "execution_error", stop.error or "", "", total_steps

            if stop.reason != StopReason.FEEDBACK_POINT:
                msg = f"unexpected stop reason: {stop.reason.value}"
                return "execution_error", msg, "", total_steps

            if stop.feedback_description is None or stop.feedback_prompt is None:
                return "execution_error", "feedback stop missing description or prompt", "", total_steps

            snapshot = self._env.observe()[0]

    def _verdict(self, status: RunStatus, value: str) -> bool:
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
