"""Composition root: wires env, planner, executor, and the run loop."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import config
from executor import BoundaryReason, ExecutionBoundary, Executor
from libero_env import LiberoEnv
from planner import (
    FeedbackHandoff,
    LLMPlanner,
    PlannerAction,
    PlannerDecision,
)
from skills import SkillRegistry, arm_skill_catalog, register_libero_skills

logger = logging.getLogger(__name__)


def _return_indicates_failure(value: Any) -> bool:
    return isinstance(value, str) and "failed to" in value.lower()


@dataclass
class RunResult:
    """Outcome of ``Runner.run()``."""

    status: str
    success: bool
    replans: int = 0
    steps: int = 0
    value: Any = None
    error: str | None = None


@dataclass
class _LoopState:
    replans: int = 0
    total_steps: int = 0
    status: str = "done"
    value: Any = None
    error: str | None = None


class Runner:
    """Orchestrates plan → execute → decide until done, stopped, or capped."""

    def __init__(self, goal: str, task_suite: str, task_idx: int) -> None:
        """Wire env, skills, planner, and executor for one LIBERO task."""
        self._max_replans = config.MAX_REPLANS
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

    def run(self) -> RunResult:
        """Run the task loop and return a structured outcome."""
        try:
            loop_state = self._run_loop()
            success = self._verdict(loop_state)
            self._log_status(loop_state, success)
            return RunResult(
                status=loop_state.status,
                success=success,
                replans=loop_state.replans,
                steps=loop_state.total_steps,
                value=loop_state.value,
                error=loop_state.error,
            )
        finally:
            self._env.close()

    def _run_loop(self) -> _LoopState:
        state = _LoopState()
        current_plan = self._planner.plan()
        self._executor.reset(current_plan)

        while True:
            stop_info = self._executor.run()
            state.total_steps += stop_info.steps

            if stop_info.reason == BoundaryReason.PLAN_DONE:
                state.status = "done"
                state.value = stop_info.return_value
                return state

            if stop_info.reason == BoundaryReason.STEP_BUDGET:
                state.status = "step_budget"
                return state

            if stop_info.reason == BoundaryReason.EXECUTION_ERROR:
                state.status = "execution_error"
                state.error = stop_info.error
                return state

            if stop_info.reason != BoundaryReason.FEEDBACK_POINT:
                state.status = "execution_error"
                state.error = f"unexpected stop_info reason: {stop_info.reason.value}"
                return state

            decision = self._planner_decide(stop_info, state)
            if decision is None:
                return state

            if decision.action == PlannerAction.CONTINUE:
                continue

            if decision.action == PlannerAction.STOP:
                state.status = "stopped"
                state.error = decision.reason or None
                return state

            if decision.action == PlannerAction.NEW_PLAN:
                if state.replans >= self._max_replans:
                    state.status = "replan_limit"
                    state.error = decision.reason or stop_info.feedback_description
                    return state
                state.replans += 1
                if decision.plan is None:
                    state.status = "replan_failed"
                    state.error = "planner returned NEW_PLAN without a plan"
                    return state
                self._executor.reset(decision.plan)

    def _planner_decide(
        self, stop_info: ExecutionBoundary, state: _LoopState
    ) -> PlannerDecision | None:
        if stop_info.image_bytes is None or stop_info.feedback_prompt is None:
            state.status = "execution_error"
            state.error = "feedback stop_info missing image or prompt"
            return None
        if stop_info.feedback_description is None:
            state.status = "execution_error"
            state.error = "feedback stop_info missing description"
            return None

        handoff = FeedbackHandoff(
            description=stop_info.feedback_description,
            prompt=stop_info.feedback_prompt,
            image_bytes=stop_info.image_bytes,
        )
        try:
            return self._planner.decide(handoff)
        except RuntimeError as exc:
            state.status = "replan_failed"
            state.error = str(exc)
            return None

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
        elif state.status == "stopped":
            logger.info("Task stopped: %r (feedback halted execution)", self._goal)
        elif state.status == "replan_failed":
            logger.info(
                "Task failed: %r (replan planner error after %d replan(s))",
                self._goal,
                state.replans,
            )
        elif state.status == "replan_limit":
            logger.info(
                "Task failed: %r (replan limit reached after %d replan(s))",
                self._goal,
                state.replans,
            )
        elif state.status == "step_budget":
            logger.info("Task failed: %r (executor step budget exceeded)", self._goal)
        else:
            logger.info("Task failed: %r (status=%s)", self._goal, state.status)

        summary_parts = [
            f"success={success}",
            f"plan_status={state.status}",
            f"steps={state.total_steps}",
        ]
        if state.replans:
            summary_parts.append(f"replans={state.replans}")
        if state.value is not None:
            summary_parts.append(f"last_skill={state.value!r}")
        if state.error:
            summary_parts.append(f"error={state.error}")

        logger.info("Run summary: %s", ", ".join(summary_parts))
