from __future__ import annotations

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


class Interpreter:
    def __init__(
        self,
        registry: SkillRegistry,
        feedback_evaluator: FeedbackEvaluator,
        image_fn: Callable[[], bytes],
        goal: str,
        step_budget: int = 30,
    ) -> None:
        self._registry = registry
        self._feedback_evaluator = feedback_evaluator
        self._image_fn = image_fn
        self._goal = goal
        self._step_budget = step_budget
        self._steps = 0

    def run(self, plan: Plan) -> Any:
        """Execute *plan*. Returns the value from a ReturnNode, or None."""
        ctx: dict[str, Any] = {}
        return self._exec(plan.body, ctx)

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
            result = skill(**evaluated_args)
            self._steps += 1
            if node.assign_to is not None:
                ctx[node.assign_to] = result
            return result

        if isinstance(node, FeedbackNode):
            image_bytes = self._image_fn()
            decision = self._feedback_evaluator.evaluate(image_bytes, node.description, self._goal)
            if decision == FeedbackDecision.CONTINUE:
                return None
            if decision == FeedbackDecision.STOP:
                raise StopSignal()
            if decision == FeedbackDecision.REPLAN:
                raise ReplanSignal(node.description)
            raise ValueError(f"Unsupported feedback decision: {decision!r}")

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

            left = self._eval(expr.left, ctx)
            right = self._eval(expr.right, ctx)

            if expr.op == "==":
                return left == right
            if expr.op == "!=":
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
