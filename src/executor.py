"""Plan executor: walks the AST, runs skills/VLA, and returns execution boundaries."""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import config
from plan import (
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
from skills import SkillRegistry
from vla_client import VLAClient

logger = logging.getLogger(__name__)
_STATUS_TERMS = {"success", "stopped", "failed", "failure", "done", "error"}
_ARM_SKILLS = frozenset({"arm.grab", "arm.place"})


class BoundaryReason(str, Enum):
    """Why ``Executor.run()`` stopped."""

    FEEDBACK_POINT = "feedback_point"
    PLAN_DONE = "plan_done"
    STEP_BUDGET = "step_budget"
    EXECUTION_ERROR = "execution_error"


@dataclass
class ExecutionBoundary:
    """Payload returned when plan execution pauses or finishes."""

    reason: BoundaryReason
    steps: int = 0
    feedback_description: str | None = None
    feedback_prompt: str | None = None
    image_bytes: bytes | None = None
    return_value: Any = None
    error: str | None = None


class LiberoEnvLike(Protocol):
    """Minimal environment surface required by ``Executor``."""

    def observe(self) -> tuple[bytes, bytes, list[float]]: ...

    def step(self, action: list[float]) -> bool: ...

    def success(self) -> bool: ...


class VLAClientLike(Protocol):
    """Minimal VLA client surface required by ``Executor``."""

    def predict(
        self, image: bytes, wrist: bytes, state: list[float], task: str
    ) -> list[list[float]]: ...


class ReturnSignal(Exception):
    """Raised by a return node to unwind the call stack and deliver a value."""

    def __init__(self, value: Any = None) -> None:
        super().__init__()
        self.value = value


class StepBudgetExceeded(Exception):
    """Raised when the step budget is exhausted."""


class FeedbackPause(Exception):
    """Internal: execution paused at a feedback node for runner/planner handoff."""

    def __init__(
        self,
        *,
        description: str,
        prompt: str,
        image_bytes: bytes,
    ) -> None:
        super().__init__(description)
        self.description = description
        self.prompt = prompt
        self.image_bytes = image_bytes


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


def _clean_object_name(name: str) -> str:
    return re.sub(r"_\d+$", "", name).replace("_", " ")


class Executor:
    """Executes a plan against the environment and VLA policy."""

    def __init__(
        self,
        registry: SkillRegistry,
        env: LiberoEnvLike,
        vla: VLAClientLike | None = None,
        step_budget: int = config.STEP_BUDGET,
    ) -> None:
        self._registry = registry
        self._env = env
        self._vla = vla or VLAClient()
        self._step_budget = step_budget
        self._plan: Plan | None = None
        self._ctx: dict[str, Any] = {}
        self._steps = 0
        self._last_skill_result: str | None = None
        self._continuation: Callable[[], None] | None = None

    def reset(self, plan: Plan) -> None:
        """Load a new plan and clear execution state."""
        self._plan = plan
        self._ctx = {}
        self._steps = 0
        self._last_skill_result = None
        self._continuation = None

    def run(self) -> ExecutionBoundary:
        """Run until a boundary (feedback, done, budget, or error)."""
        if self._plan is None:
            raise RuntimeError("Executor.run() called before reset(plan)")

        steps_before = self._steps
        try:
            if self._continuation is not None:
                continuation = self._continuation
                self._continuation = None
                continuation()
            else:
                self._exec(self._plan.body, self._ctx)
            return ExecutionBoundary(
                reason=BoundaryReason.PLAN_DONE,
                steps=self._steps - steps_before,
                return_value=None,
            )
        except ReturnSignal as sig:
            return ExecutionBoundary(
                reason=BoundaryReason.PLAN_DONE,
                steps=self._steps - steps_before,
                return_value=sig.value,
            )
        except FeedbackPause as pause:
            return ExecutionBoundary(
                reason=BoundaryReason.FEEDBACK_POINT,
                steps=self._steps - steps_before,
                feedback_description=pause.description,
                feedback_prompt=pause.prompt,
                image_bytes=pause.image_bytes,
            )
        except StepBudgetExceeded:
            return ExecutionBoundary(
                reason=BoundaryReason.STEP_BUDGET,
                steps=self._steps - steps_before,
            )
        except Exception as exc:
            logger.exception("Executor execution error")
            return ExecutionBoundary(
                reason=BoundaryReason.EXECUTION_ERROR,
                steps=self._steps - steps_before,
                error=str(exc),
            )

    def _append_continuation(self, fn: Callable[[], None]) -> None:
        previous = self._continuation
        if previous is None:
            self._continuation = fn
            return

        def chained() -> None:
            previous()
            fn()

        self._continuation = chained

    def _exec_steps(self, steps: list[StmtNode], ctx: dict[str, Any]) -> Any:
        last: Any = None
        for index, step in enumerate(steps):
            try:
                last = self._exec(step, ctx)
            except FeedbackPause:
                remaining = steps[index + 1 :]
                if remaining:

                    def continue_steps(
                        remaining_steps: list[StmtNode] = remaining,
                        context: dict[str, Any] = ctx,
                    ) -> None:
                        self._exec_steps(remaining_steps, context)

                    self._append_continuation(continue_steps)
                raise
        return last

    def _manipulate(
        self,
        skill: str,
        object_name: str,
        location: str | None = None,
    ) -> dict[str, Any]:
        cleaned = _clean_object_name(object_name)
        if skill == "arm.grab":
            task = f"pick up the {cleaned}"
        elif skill == "arm.place":
            task = f"put the {cleaned} in the {location}"
        else:
            raise ValueError(f"unsupported arm skill: {skill!r}")

        skill_steps = 0
        while not self._env.success() and skill_steps < config.MAX_SKILL_STEPS:
            image, wrist, state = self._env.observe()
            chunk = self._vla.predict(image, wrist, state, task)
            for action in chunk[: config.REPLAN_STEPS]:
                self._env.step(action)
                skill_steps += 1
                if self._env.success() or skill_steps >= config.MAX_SKILL_STEPS:
                    break

        if self._env.success():
            return {"status": "success"}
        return {"status": "failed", "steps": skill_steps}

    def _exec(self, node: StmtNode, ctx: dict[str, Any]) -> Any:
        if isinstance(node, SeqNode):
            return self._exec_steps(node.steps, ctx)

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
                    raise RuntimeError(
                        f"while loop exceeded max_iterations={node.max_iterations}"
                    )
                self._exec(node.body, ctx)
                iterations += 1
            return None

        if isinstance(node, CallNode):
            if self._steps >= self._step_budget:
                raise StepBudgetExceeded()
            evaluated_args = {
                key: self._eval(value, ctx) for key, value in node.args.items()
            }
            args_str = ", ".join(f"{k}={v!r}" for k, v in evaluated_args.items())
            logger.info("Uses skill: %s(%s)", node.skill, args_str)

            if node.skill in _ARM_SKILLS:
                object_name = str(evaluated_args.get("object_name", ""))
                location = evaluated_args.get("location")
                location_str = str(location) if location is not None else None
                result = self._manipulate(node.skill, object_name, location_str)
            else:
                skill = self._registry.get(node.skill)
                result = skill(**evaluated_args)

            self._last_skill_result = str(result)
            logger.info("Skill result (%s): %r", node.skill, result)
            self._steps += 1
            if node.assign_to is not None:
                ctx[node.assign_to] = result
            return result

        if isinstance(node, FeedbackNode):
            image_bytes = self._env.observe()[0]
            prompt = node.description
            if self._last_skill_result:
                prompt = (
                    f"{prompt}\n\nPrior skill result: {self._last_skill_result}"
                )
            logger.info("Feedback checkpoint: %r", node.description)
            raise FeedbackPause(
                description=node.description,
                prompt=prompt,
                image_bytes=image_bytes,
            )

        if isinstance(node, ReturnNode):
            value = self._eval(node.value, ctx) if node.value is not None else None
            raise ReturnSignal(value=value)

        raise TypeError(f"Unsupported statement node: {type(node)!r}")

    def _eval(self, expr: Expr, ctx: dict[str, Any]) -> Any:
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
                raise TypeError(
                    f"Incomparable operands for {expr.op}: {left!r}, {right!r}"
                ) from exc

            raise ValueError(f"Unsupported binary operator: {expr.op!r}")

        if isinstance(expr, NotExpr):
            return not self._eval(expr.operand, ctx)

        raise TypeError(f"Unsupported expression node: {type(expr)!r}")
