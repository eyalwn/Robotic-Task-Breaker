from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import config
from executor import ExecutionResult, Executor
from libero_env import LiberoEnv
from planner import LLMFeedbackEvaluator, LLMPlanner
from plan import CallNode, IfNode, Plan, SeqNode, StmtNode, WhileNode
from skills import SkillRegistry, arm_skill_catalog, register_libero_skills
from vla_client import VLAClient

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


def _return_indicates_failure(value: Any) -> bool:
    return isinstance(value, str) and "failed to" in value.lower()


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
        goal: str,
        task_suite: str = "libero_10",
        task_idx: int = 0,
    ) -> None:
        self._max_replans = config.MAX_REPLANS
        self._step_budget = config.STEP_BUDGET
        self._env = LiberoEnv(task_suite, task_idx)
        self._vla = VLAClient()
        self._planner = LLMPlanner()
        self._feedback = LLMFeedbackEvaluator()
        self._registry = SkillRegistry()
        register_libero_skills(self._registry, self._env)
        self._enriched_goal = self._enrich_goal(goal, self._registry)
        self._catalog = self._registry.render_catalog() + "\n" + arm_skill_catalog()
        self._executor = self._new_executor(self._enriched_goal)

    def _new_executor(self, goal: str) -> Executor:
        return Executor(
            self._registry,
            self._feedback,
            image_fn=lambda: self._env.observe()[0],
            goal=goal,
            env=self._env,
            vla=self._vla,
            step_budget=self._step_budget,
        )

    def run(self) -> RunResult:
        try:
            exec_result = self._run_loop()
            success = self._verdict(exec_result)
            self._log_status(exec_result, success)
            return RunResult(
                status=exec_result.status,
                success=success,
                replans=exec_result.replans,
                steps=exec_result.steps,
                value=exec_result.value,
                error=exec_result.error,
            )
        finally:
            self._env.close()

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

    def _run_loop(self) -> ExecutionResult:
        replans = 0
        current_plan = self._planner.plan(self._enriched_goal, self._catalog)
        total_steps = 0

        while True:
            self._executor = self._new_executor(self._enriched_goal)
            result = self._executor.run(current_plan)

            if result.status == "replan":
                total_steps += result.steps
                if replans >= self._max_replans:
                    return ExecutionResult(
                        status="replan_limit",
                        replans=replans,
                        steps=total_steps,
                        error=result.error,
                    )
                replans += 1
                try:
                    current_plan = self._planner.plan(
                        self._enriched_goal,
                        self._catalog,
                        context=result.error or "",
                    )
                except RuntimeError as exc:
                    return ExecutionResult(
                        status="replan_failed",
                        replans=replans,
                        steps=total_steps,
                        error=str(exc),
                    )
                continue

            result.replans = replans
            result.steps = total_steps + result.steps
            return result

    def _verdict(self, result: ExecutionResult) -> bool:
        if result.status != "done":
            return False
        if _return_indicates_failure(result.value):
            return False
        return bool(self._env.success())

    def _log_status(self, result: ExecutionResult, success: bool) -> None:
        headline = self._enriched_goal.split("\n")[0]

        if success:
            logger.info("Task succeeded: %r", headline)
        elif result.status == "done":
            if _return_indicates_failure(result.value):
                logger.info(
                    "Task not achieved: %r (plan finished; skill reported failure)",
                    headline,
                )
            else:
                logger.info(
                    "Task not achieved: %r (plan finished; env goal check failed)",
                    headline,
                )
        elif result.status == "stopped":
            logger.info("Task stopped: %r (feedback halted execution)", headline)
        elif result.status == "replan_failed":
            logger.info(
                "Task failed: %r (replan planner error after %d replan(s))",
                headline,
                result.replans,
            )
        elif result.status == "replan_limit":
            logger.info(
                "Task failed: %r (replan limit reached after %d replan(s))",
                headline,
                result.replans,
            )
        elif result.status == "step_budget":
            logger.info("Task failed: %r (executor step budget exceeded)", headline)
        else:
            logger.info("Task failed: %r (status=%s)", headline, result.status)

        summary_parts = [
            f"success={success}",
            f"plan_status={result.status}",
            f"steps={result.steps}",
        ]
        if result.replans:
            summary_parts.append(f"replans={result.replans}")
        if result.value is not None:
            summary_parts.append(f"last_skill={result.value!r}")
        if result.error:
            summary_parts.append(f"error={result.error}")

        logger.info("Run summary: %s", ", ".join(summary_parts))
