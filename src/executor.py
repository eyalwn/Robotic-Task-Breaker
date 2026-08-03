"""Plan executor: walks the AST, runs skills/VLA, and returns execution boundaries."""
from __future__ import annotations

import json
import logging
import re
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
    """Why ``Executor.run(plan)`` stopped."""

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

    def run(self, plan: Plan) -> ExecutionBoundary:
        """Execute plan from scratch until feedback, done, budget, or error."""
        ctx: dict[str, Any] = {}
        run_steps: list[int] = [0]
        last_skill_result: list[str | None] = [None]
        try:
            self._exec(plan.body, ctx, run_steps, last_skill_result)
            return ExecutionBoundary(
                reason=BoundaryReason.PLAN_DONE,
                steps=run_steps[0],
                return_value=None,
            )
        except ReturnSignal as sig:
            return ExecutionBoundary(
                reason=BoundaryReason.PLAN_DONE,
                steps=run_steps[0],
                return_value=sig.value,
            )
        except FeedbackPause as pause:
            return ExecutionBoundary(
                reason=BoundaryReason.FEEDBACK_POINT,
                steps=run_steps[0],
                feedback_description=pause.description,
                feedback_prompt=pause.prompt,
                image_bytes=pause.image_bytes,
            )
        except StepBudgetExceeded:
            return ExecutionBoundary(
                reason=BoundaryReason.STEP_BUDGET,
                steps=run_steps[0],
            )
        except Exception as exc:
            logger.exception("Executor execution error")
            return ExecutionBoundary(
                reason=BoundaryReason.EXECUTION_ERROR,
                steps=run_steps[0],
                error=str(exc),
            )

    def _exec_steps(
        self,
        steps: list[StmtNode],
        ctx: dict[str, Any],
        run_steps: list[int],
        last_skill_result: list[str | None],
    ) -> Any:
        last: Any = None
        for step in steps:
            last = self._exec(step, ctx, run_steps, last_skill_result)
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

    def _exec(
        self,
        node: StmtNode,
        ctx: dict[str, Any],
        run_steps: list[int],
        last_skill_result: list[str | None],
    ) -> Any:
        if isinstance(node, SeqNode):
            return self._exec_steps(node.steps, ctx, run_steps, last_skill_result)

        if isinstance(node, IfNode):
            if self._eval(node.condition, ctx):
                return self._exec(node.then_branch, ctx, run_steps, last_skill_result)
            if node.else_branch is not None:
                return self._exec(node.else_branch, ctx, run_steps, last_skill_result)
            return None

        if isinstance(node, WhileNode):
            iterations = 0
            while self._eval(node.condition, ctx):
                if iterations >= node.max_iterations:
                    raise RuntimeError(
                        f"while loop exceeded max_iterations={node.max_iterations}"
                    )
                self._exec(node.body, ctx, run_steps, last_skill_result)
                iterations += 1
            return None

        if isinstance(node, CallNode):
            if run_steps[0] >= self._step_budget:
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

            last_skill_result[0] = str(result)
            logger.info("Skill result (%s): %r", node.skill, result)
            run_steps[0] += 1
            if node.assign_to is not None:
                ctx[node.assign_to] = result
            return result

        if isinstance(node, FeedbackNode):
            image_bytes = self._env.observe()[0]
            prompt = node.description
            if last_skill_result[0]:
                prompt = (
                    f"{prompt}\n\nPrior skill result: {last_skill_result[0]}"
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
