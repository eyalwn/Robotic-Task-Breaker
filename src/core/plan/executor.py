from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from core.skills import SkillRegistry

from .feedback import FeedbackDecision, FeedbackEvaluator
from .schema import (
    BinOpExpr,
    CallNode,
    Expr,
    FeedbackNode,
    IfNode,
    LiteralExpr,
    NotExpr,
    Plan,
    ReturnNode,
    SeqNode,
    StmtNode,
    VarExpr,
    WhileNode,
)


logger = logging.getLogger(__name__)
_STATUS_TERMS = {"success", "stopped", "failed", "failure", "done", "error"}


def _status_from_value(value: Any) -> str | None:
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, str):
            return status.strip().lower()
        return None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    low = text.lower()
    if low in _STATUS_TERMS:
        return low
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    status = parsed.get("status")
    if isinstance(status, str):
        return status.strip().lower()
    return None


def _coerce_status_comparison(left: Any, right: Any) -> tuple[Any, Any]:
    left_status = _status_from_value(left)
    right_status = _status_from_value(right)
    if left_status is None or right_status is None:
        return left, right
    return left_status, right_status


def _log_feedback(node_description: str, prompt: str, outcome: FeedbackOutcome) -> None:
    logger.info("Feedback LLM asked: %r", node_description)
    if prompt != node_description:
        logger.debug("Feedback LLM prompt (full): %s", prompt)
    if outcome.assessment is not None:
        if outcome.reason:
            logger.info(
                "Feedback LLM response: %s — %s",
                outcome.assessment,
                outcome.reason,
            )
        else:
            logger.info("Feedback LLM response: %s", outcome.assessment)
    else:
        logger.info("Feedback LLM: not consulted (stub evaluator)")
    logger.info(
        "Feedback executor decision: %s",
        outcome.decision.value.upper(),
    )


class ReplanSignal(Exception):
    """Raised by a feedback node when the LLM evaluator returns REPLAN."""

    def __init__(self, description: str) -> None:
        super().__init__(description)
        self.description = description


class ReturnSignal(Exception):
    """Raised by a return node to unwind the call stack and deliver a value."""

    def __init__(self, value: Any = None) -> None:
        super().__init__()
        self.value = value


class StopSignal(Exception):
    """Raised by a feedback node when the LLM evaluator returns STOP."""


class StepBudgetExceeded(Exception):
    """Raised when the step budget is exhausted."""


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
        feedback_evaluator: FeedbackEvaluator,
        image_fn: Callable[[], bytes],
        goal: str,
        step_budget: int = 30,
        trace: "TraceWriter | None" = None,
    ) -> None:
        self._registry = registry
        self._feedback_evaluator = feedback_evaluator
        self._image_fn = image_fn
        self._goal = goal
        self._step_budget = step_budget
        self._steps = 0
        self._trace = trace
        self._last_skill_result: str | None = None

    def run(self, plan: Plan) -> ExecutionResult:
        """Execute *plan* and return an ExecutionResult (signals caught internally)."""
        ctx: dict[str, Any] = {}
        try:
            self._exec(plan.body, ctx)
            return ExecutionResult(status="done", steps=self._steps, value=None)
        except ReturnSignal as sig:
            return ExecutionResult(status="done", steps=self._steps, value=sig.value)
        except StopSignal:
            return ExecutionResult(status="stopped", steps=self._steps)
        except ReplanSignal as sig:
            return ExecutionResult(status="replan", steps=self._steps, error=sig.description)
        except StepBudgetExceeded:
            return ExecutionResult(status="step_budget", steps=self._steps)

    def _exec(self, node: StmtNode, ctx: dict[str, Any]) -> Any:
        """Dispatch on node kind."""
        if isinstance(node, SeqNode):
            last: Any = None
            for step in node.steps:
                last = self._exec(step, ctx)
            return last

        if isinstance(node, IfNode):
            if self._eval(node.condition, ctx):
                return self._exec(node.then_branch, ctx)
            if node.else_branch is not None:
                return self._exec(node.else_branch, ctx)
            return None

        if isinstance(node, WhileNode):
            iterations = 0
            while self._eval(node.condition, ctx):
                if iterations >= node.max_iterations:
                    raise RuntimeError(f"while loop exceeded max_iterations={node.max_iterations}")
                self._exec(node.body, ctx)
                iterations += 1
            return None

        if isinstance(node, CallNode):
            if self._steps >= self._step_budget:
                raise StepBudgetExceeded()
            skill = self._registry.get(node.skill)
            evaluated_args = {key: self._eval(value, ctx) for key, value in node.args.items()}
            args_str = ", ".join(f"{k}={v!r}" for k, v in evaluated_args.items())
            logger.info("Uses skill: %s(%s)", node.skill, args_str)
            result = skill(**evaluated_args)
            self._last_skill_result = str(result)
            logger.info("Skill result (%s): %r", node.skill, result)
            self._steps += 1
            if node.assign_to is not None:
                ctx[node.assign_to] = result
            if self._trace:
                self._trace.write(
                    {
                        "event": "skill_call",
                        "skill": node.skill,
                        "args": {k: str(v) for k, v in evaluated_args.items()},
                        "result": str(result),
                        "step": self._steps,
                    }
                )
            return result

        if isinstance(node, FeedbackNode):
            image_bytes = self._image_fn()
            description = node.description
            if self._last_skill_result:
                description = (
                    f"{description}\n\nPrior skill result: {self._last_skill_result}"
                )
            outcome = self._feedback_evaluator.evaluate(
                image_bytes, description, self._goal
            )
            _log_feedback(node.description, description, outcome)
            if self._trace:
                event: dict[str, Any] = {
                    "event": "feedback",
                    "description": node.description,
                    "decision": outcome.decision.value,
                    "step": self._steps,
                }
                if outcome.reason:
                    event["reason"] = outcome.reason
                self._trace.write(event)
            if outcome.decision == FeedbackDecision.CONTINUE:
                return None
            if outcome.decision == FeedbackDecision.STOP:
                raise StopSignal()
            if outcome.decision == FeedbackDecision.REPLAN:
                raise ReplanSignal(node.description)
            raise ValueError(f"Unsupported feedback decision: {outcome.decision!r}")

        if isinstance(node, ReturnNode):
            value = self._eval(node.value, ctx) if node.value is not None else None
            raise ReturnSignal(value=value)

        raise TypeError(f"Unsupported statement node: {type(node)!r}")

    def _eval(self, expr: Expr, ctx: dict[str, Any]) -> Any:
        """Evaluate an expression. Only literals, vars, binop, not allowed.
        Raises NameError for unbound variables.
        Raises TypeError for invalid operand types.
        """
        if isinstance(expr, LiteralExpr):
            return expr.value

        if isinstance(expr, VarExpr):
            if expr.name not in ctx:
                raise NameError(f"Variable {expr.name!r} is not bound")
            return ctx[expr.name]

        if isinstance(expr, BinOpExpr):
            if expr.op == "and":
                left = self._eval(expr.left, ctx)
                if not left:
                    return left
                return self._eval(expr.right, ctx)

            if expr.op == "or":
                left = self._eval(expr.left, ctx)
                if left:
                    return left
                return self._eval(expr.right, ctx)

            if expr.op == "in":
                left = self._eval(expr.left, ctx)
                right = self._eval(expr.right, ctx)
                if right is None:
                    return False
                return left in right

            left = self._eval(expr.left, ctx)
            right = self._eval(expr.right, ctx)

            if expr.op == "==":
                left, right = _coerce_status_comparison(left, right)
                return left == right
            if expr.op == "!=":
                left, right = _coerce_status_comparison(left, right)
                return left != right

            try:
                if expr.op == "<":
                    return left < right
                if expr.op == "<=":
                    return left <= right
                if expr.op == ">":
                    return left > right
                if expr.op == ">=":
                    return left >= right
            except TypeError as exc:
                raise TypeError(f"Incomparable operands for {expr.op}: {left!r}, {right!r}") from exc

            raise ValueError(f"Unsupported binary operator: {expr.op!r}")

        if isinstance(expr, NotExpr):
            return not self._eval(expr.operand, ctx)

        raise TypeError(f"Unsupported expression node: {type(expr)!r}")
