from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.skills import SkillRegistry

from .feedback import FeedbackEvaluator
from .interpreter import Interpreter, ReplanSignal, ReturnSignal, StepBudgetExceeded, StopSignal
from .planner import Planner
from .schema import Plan


@dataclass
class ExecutionResult:
    status: str
    replans: int = 0
    steps: int = 0
    value: Any = None
    error: str | None = None


class Executor:
    def __init__(
        self,
        registry: SkillRegistry,
        planner: Planner,
        feedback_evaluator: FeedbackEvaluator,
        image_fn: Callable[[], bytes],
        max_replans: int = 3,
        step_budget: int = 30,
    ) -> None:
        self._registry = registry
        self._planner = planner
        self._feedback_evaluator = feedback_evaluator
        self._image_fn = image_fn
        self._max_replans = max_replans
        self._step_budget = step_budget

    def run(self, goal: str, plan: Plan) -> ExecutionResult:
        """Run *plan* toward *goal*, replanning up to *max_replans* times."""
        replans = 0
        current_plan = plan
        total_steps = 0
        catalog = self._registry.render_catalog()

        while True:
            interp = Interpreter(
                self._registry,
                self._feedback_evaluator,
                self._image_fn,
                goal=goal,
                step_budget=self._step_budget,
            )
            try:
                interp.run(current_plan)
                return ExecutionResult(
                    status="done",
                    replans=replans,
                    steps=total_steps + interp._steps,
                    value=None,
                )
            except ReturnSignal as sig:
                return ExecutionResult(
                    status="done",
                    replans=replans,
                    steps=total_steps + interp._steps,
                    value=sig.value,
                )
            except StopSignal:
                return ExecutionResult(
                    status="stopped",
                    replans=replans,
                    steps=total_steps + interp._steps,
                )
            except ReplanSignal as sig:
                total_steps += interp._steps
                if replans >= self._max_replans:
                    return ExecutionResult(
                        status="replan_limit",
                        replans=replans,
                        steps=total_steps,
                        error=sig.description,
                    )
                replans += 1
                current_plan = self._planner.plan(goal, catalog, context=sig.description)
            except StepBudgetExceeded:
                return ExecutionResult(
                    status="step_budget",
                    replans=replans,
                    steps=total_steps + interp._steps,
                )
