"""Composition root: wires env, planner, executor, and the run loop."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from executor import BoundaryReason, ExecutionBoundary, Executor
from libero_env import LiberoEnv
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

@dataclass
class _LoopState:
    total_steps: int = 0
    status: str = "done"
    value: Any = None
    error: str | None = None


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
        try:
            state = self._run_loop()
        except RuntimeError as exc:
            state = _LoopState(status="plan_failed", error=str(exc))
        finally:
            self._env.close()
        success = self._verdict(state)
        self._log_status(state, success)

    def _run_loop(self) -> _LoopState:
        state = _LoopState()
        progress_history: list[str] = []

        result = self._planner.plan(None)
        if result.stop or result.plan is None:
            state.status = "plan_failed"
            state.error = result.reason or "initial plan missing"
            return state

        current_plan = result.plan

        while True:
            boundary = self._executor.run(current_plan)
            state.total_steps += boundary.steps
            progress_history.append(_history_line(boundary))

            if boundary.reason == BoundaryReason.PLAN_DONE:
                state.status = "done"
                state.value = boundary.return_value
                return state

            if boundary.reason == BoundaryReason.STEP_BUDGET:
                state.status = "step_budget"
                return state

            if boundary.reason == BoundaryReason.EXECUTION_ERROR:
                state.status = "execution_error"
                state.error = boundary.error
                return state

            if boundary.reason != BoundaryReason.FEEDBACK_POINT:
                state.status = "execution_error"
                state.error = f"unexpected boundary reason: {boundary.reason.value}"
                return state

            handoff = self._build_handoff(boundary, state, progress_history)
            if handoff is None:
                return state

            result = self._planner.plan(handoff)
            if result.stop:
                state.status = "stopped"
                state.error = result.reason or None
                return state

            if result.plan is None:
                state.status = "plan_failed"
                state.error = "planner returned no plan"
                return state

            current_plan = result.plan

    def _build_handoff(
        self,
        boundary: ExecutionBoundary,
        state: _LoopState,
        progress_history: list[str],
    ) -> PlanningHandoff | None:
        if boundary.image_bytes is None or boundary.feedback_prompt is None:
            state.status = "execution_error"
            state.error = "feedback boundary missing image or prompt"
            return None
        if boundary.feedback_description is None:
            state.status = "execution_error"
            state.error = "feedback boundary missing description"
            return None

        return PlanningHandoff(
            description=boundary.feedback_description,
            prompt=boundary.feedback_prompt,
            image_bytes=boundary.image_bytes,
            progress_history=tuple(progress_history),
        )

    def _verdict(self, state: _LoopState) -> bool:
        if state.status != "done":
            return False
        if _return_indicates_failure(state.value):
            return False
        return bool(self._env.success())

    def _log_status(self, state: _LoopState, success: bool) -> None:
        if success:
            logger.info("Task succeeded: %r", self._goal)
        elif state.status == "done":
            if _return_indicates_failure(state.value):
                logger.info(
                    "Task not achieved: %r (plan finished; skill reported failure)",
                    self._goal,
                )
            else:
                logger.info(
                    "Task not achieved: %r (plan finished; env goal check failed)",
                    self._goal,
                )
        elif state.status == "plan_failed":
            logger.info("Task failed: %r (planner error during replan)", self._goal)
        elif state.status == "stopped":
            logger.info("Task stopped: %r (%s)", self._goal, state.error)
        elif state.status == "step_budget":
            logger.info("Task failed: %r (executor step budget exceeded)", self._goal)
        else:
            logger.info("Task failed: %r (status=%s)", self._goal, state.status)

        summary_parts = [
            f"success={success}",
            f"plan_status={state.status}",
            f"steps={state.total_steps}",
        ]
        if state.value is not None:
            summary_parts.append(f"last_skill={state.value!r}")
        if state.error:
            summary_parts.append(f"error={state.error}")

        logger.info("Run summary: %s", ", ".join(summary_parts))
