"""Composition root: wires env, planner, executor, and the run loop."""
from __future__ import annotations

import logging
from typing import Any

from executor import BoundaryReason, ExecutionBoundary, Executor
from libero_env import LiberoEnv
from plan import planner_stop_reason, validate_plan
from planner import LLMPlanner, PlanningHandoff
from skills import SkillRegistry, arm_skill_catalog, register_libero_skills

logger = logging.getLogger(__name__)


def _return_indicates_failure(value: Any) -> bool:
    return isinstance(value, str) and "failed to" in value.lower()


def _history_line(boundary: ExecutionBoundary) -> str:
    reason = boundary.reason.value
    if boundary.reason == BoundaryReason.FEEDBACK_POINT:
        detail = boundary.feedback_description or ""
    elif boundary.reason == BoundaryReason.PLAN_DONE:
        detail = str(boundary.return_value) if boundary.return_value is not None else ""
    elif boundary.reason == BoundaryReason.EXECUTION_ERROR:
        detail = boundary.error or ""
    else:
        detail = ""
    return f"{reason}: {detail} | steps={boundary.steps}"


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
        status = "done"
        error: str | None = None
        value: Any = None
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

    def _run_loop(self) -> tuple[str, str | None, Any, int]:
        total_steps = 0
        progress_history: list[str] = []

        plan_dict = self._planner.plan(None)
        reason = planner_stop_reason(plan_dict)
        if reason:
            return "stopped", reason, None, total_steps

        current_plan = validate_plan(plan_dict)

        while True:
            boundary = self._executor.run(current_plan)
            total_steps += boundary.steps
            progress_history.append(_history_line(boundary))

            if boundary.reason == BoundaryReason.PLAN_DONE:
                return "done", None, boundary.return_value, total_steps

            if boundary.reason == BoundaryReason.STEP_BUDGET:
                return "step_budget", None, None, total_steps

            if boundary.reason == BoundaryReason.EXECUTION_ERROR:
                return "execution_error", boundary.error, None, total_steps

            if boundary.reason != BoundaryReason.FEEDBACK_POINT:
                msg = f"unexpected boundary reason: {boundary.reason.value}"
                return "execution_error", msg, None, total_steps

            handoff, handoff_error = self._build_handoff(boundary, progress_history)
            if handoff is None:
                return "execution_error", handoff_error, None, total_steps

            plan_dict = self._planner.plan(handoff)
            reason = planner_stop_reason(plan_dict)
            if reason:
                return "stopped", reason, None, total_steps

            current_plan = validate_plan(plan_dict)

    def _build_handoff(
        self,
        boundary: ExecutionBoundary,
        progress_history: list[str],
    ) -> tuple[PlanningHandoff | None, str | None]:
        if boundary.image_bytes is None or boundary.feedback_prompt is None:
            return None, "feedback boundary missing image or prompt"
        if boundary.feedback_description is None:
            return None, "feedback boundary missing description"

        return PlanningHandoff(
            description=boundary.feedback_description,
            prompt=boundary.feedback_prompt,
            image_bytes=boundary.image_bytes,
            progress_history=tuple(progress_history),
        ), None

    def _verdict(self, status: str, value: Any) -> bool:
        if status != "done":
            return False
        if _return_indicates_failure(value):
            return False
        return bool(self._env.success())

    def _log_status(
        self,
        status: str,
        success: bool,
        value: Any,
        error: str | None,
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
        if value is not None:
            summary_parts.append(f"last_skill={value!r}")
        if error:
            summary_parts.append(f"error={error}")

        logger.info("Run summary: %s", ", ".join(summary_parts))
