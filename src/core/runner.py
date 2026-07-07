from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from core.backend import Backend
from core.plan.executor import ExecutionResult, Executor
from core.plan.feedback import FeedbackEvaluator
from core.plan.planner import Planner
from core.plan.schema import CallNode, IfNode, Plan, SeqNode, StmtNode, WhileNode
from core.plan.trace import TraceWriter
from core.skills import SkillRegistry


logger = logging.getLogger(__name__)


def _collect_plan_skills(node: StmtNode) -> list[str]:
    if isinstance(node, CallNode):
        return [node.skill]
    if isinstance(node, SeqNode):
        skills: list[str] = []
        for step in node.steps:
            skills.extend(_collect_plan_skills(step))
        return skills
    if isinstance(node, IfNode):
        skills = _collect_plan_skills(node.then_branch)
        if node.else_branch is not None:
            skills.extend(_collect_plan_skills(node.else_branch))
        return skills
    if isinstance(node, WhileNode):
        return _collect_plan_skills(node.body)
    return []


def _write_plan_start(
    trace,
    goal: str,
    plan: Plan,
    *,
    replan: int | None = None,
) -> None:
    skills = _collect_plan_skills(plan.body)
    event: dict[str, Any] = {
        "event": "plan_start",
        "goal": goal,
        "plan": plan.model_dump(),
    }
    if skills:
        event["skills"] = skills
    if replan is not None:
        event["replan"] = replan
    if trace:
        trace.write(event)


def _return_indicates_failure(value: Any) -> bool:
    return isinstance(value, str) and "failed to" in value.lower()


def _write_plan_end(trace, result: ExecutionResult) -> None:
    if trace:
        trace.write(
            {
                "event": "plan_end",
                "status": result.status,
                "replans": result.replans,
                "steps": result.steps,
                "value": str(result.value),
            }
        )


def _finish_replan_limit(
    trace,
    replans: int,
    total_steps: int,
    error: str | None,
) -> ExecutionResult:
    result = ExecutionResult(
        status="replan_limit",
        replans=replans,
        steps=total_steps,
        error=error,
    )
    _write_plan_end(trace, result)
    return result


def _finish_replan_failed(
    trace,
    replans: int,
    total_steps: int,
    exc: RuntimeError,
) -> ExecutionResult:
    result = ExecutionResult(
        status="replan_failed",
        replans=replans,
        steps=total_steps,
        error=str(exc),
    )
    if trace:
        trace.write(
            {
                "event": "replan_failed",
                "attempt": replans,
                "error": str(exc),
            }
        )
    _write_plan_end(trace, result)
    return result


@dataclass
class RunResult:
    status: str
    success: bool
    replans: int = 0
    steps: int = 0
    value: Any = None
    error: str | None = None


class Runner:
    def __init__(
        self,
        model: str | None = None,
        max_replans: int = 3,
        step_budget: int = 30,
        trace_dir: str | None = None,
    ) -> None:
        self._model = model if model is not None else os.environ.get("LLM_MODEL")
        if max_replans == 3:
            self._max_replans = int(os.environ.get("LONGOR_MAX_REPLANS", "3"))
        else:
            self._max_replans = max_replans
        if step_budget == 30:
            self._step_budget = int(os.environ.get("LONGOR_STEP_BUDGET", "30"))
        else:
            self._step_budget = step_budget
        self._trace_dir = (
            trace_dir if trace_dir is not None else os.environ.get("LONGOR_TRACE_DIR")
        )

    def run(
        self,
        backend: Backend,
        goal: str,
        planner: Planner | None = None,
        feedback: FeedbackEvaluator | None = None,
    ) -> RunResult:
        registry = SkillRegistry()
        backend.register_skills(registry, goal)
        enriched = self._enrich_goal(goal, registry)
        planner = planner or self._default_planner()
        feedback = feedback or self._default_feedback()

        trace: TraceWriter | None = (
            TraceWriter(self._trace_dir, enriched) if self._trace_dir else None
        )
        try:
            exec_result = self._run_loop(
                registry, planner, feedback, backend, enriched, trace
            )
            success = self._verdict(exec_result, backend)
            self._log_status(exec_result, success, enriched)
            return RunResult(
                status=exec_result.status,
                success=success,
                replans=exec_result.replans,
                steps=exec_result.steps,
                value=exec_result.value,
                error=exec_result.error,
            )
        finally:
            if trace:
                trace.close()
            try:
                backend.close()
            except Exception:
                pass

    def _default_planner(self) -> Planner:
        from core.plan.llm_planner import LLMPlanner

        return LLMPlanner(model=self._model, max_plan_retries=3)

    def _default_feedback(self) -> FeedbackEvaluator:
        from core.plan.llm_feedback import LLMFeedbackEvaluator

        return LLMFeedbackEvaluator(model=self._model)

    def _enrich_goal(self, goal: str, registry: SkillRegistry) -> str:
        try:
            objects = registry.get("state.list_objects")()
        except Exception:
            objects = None
        logger.info("Scene objects: %s", objects)
        if objects:
            return (
                goal
                + "\n\nAvailable scene objects (use these exact names in all skill args): "
                + ", ".join(str(o) for o in objects)
            )
        return goal

    def _run_loop(
        self,
        registry: SkillRegistry,
        planner: Planner,
        feedback: FeedbackEvaluator,
        backend: Backend,
        goal: str,
        trace: TraceWriter | None,
    ) -> ExecutionResult:
        replans = 0
        catalog = registry.render_catalog()
        current_plan = planner.plan(goal, catalog)

        if trace:
            _write_plan_start(trace, goal, current_plan)

        total_steps = 0
        while True:
            executor = Executor(
                registry,
                feedback,
                backend.capture_image,
                goal=goal,
                step_budget=self._step_budget,
                trace=trace,
            )
            result = executor.run(current_plan)

            if result.status == "replan":
                total_steps += result.steps
                if replans >= self._max_replans:
                    return _finish_replan_limit(
                        trace, replans, total_steps, result.error
                    )
                if trace:
                    trace.write(
                        {"event": "replan", "attempt": replans + 1, "reason": result.error}
                    )
                replans += 1
                try:
                    current_plan = planner.plan(goal, catalog, context=result.error or "")
                except RuntimeError as exc:
                    return _finish_replan_failed(trace, replans, total_steps, exc)
                if trace:
                    _write_plan_start(trace, goal, current_plan, replan=replans)
                continue

            result.replans = replans
            result.steps = total_steps + result.steps
            _write_plan_end(trace, result)
            return result

    def _verdict(self, result: ExecutionResult, backend: Backend) -> bool:
        if result.status != "done":
            return False
        if _return_indicates_failure(result.value):
            return False
        return bool(backend.success())

    def _log_status(self, result: ExecutionResult, success: bool, goal: str) -> None:
        headline = goal.split("\n")[0]
        if result.status == "done":
            logger.info("Task COMPLETED: %r", headline)
        else:
            logger.warning("Task FAILED (status=%s): %r", result.status, headline)

        if result.status == "replan_failed":
            logger.error(
                "Run stopped: replan failed after %d replan(s), %d steps — %s",
                result.replans,
                result.steps,
                result.error,
            )
        elif result.status != "done":
            logger.warning(
                "Run finished with status=%s replans=%d steps=%d error=%r",
                result.status,
                result.replans,
                result.steps,
                result.error,
            )
        elif result.value is not None and isinstance(result.value, str):
            logger.info(
                "Run finished with return value=%r (status=%s replans=%d steps=%d)",
                result.value,
                result.status,
                result.replans,
                result.steps,
            )
        else:
            logger.info(
                "Run complete: status=%s replans=%d steps=%d value=%r",
                result.status,
                result.replans,
                result.steps,
                result.value,
            )

        if result.status == "done" and not success:
            if _return_indicates_failure(result.value):
                logger.error(
                    "Run status=done but return value indicates failure: %r",
                    result.value,
                )
            else:
                logger.error(
                    "Run status=done but backend success is false — task not achieved"
                )
