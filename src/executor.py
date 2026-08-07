"""Execute plan AST nodes against LIBERO and return stop-aware summaries."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

import config
import plan as plan_ast
from libero_env import LiberoEnv
from skills import SkillRegistry
from vla_client import VLAClient

logger = logging.getLogger(__name__)
_ARM_SKILLS = frozenset({"arm.grab", "arm.place"})


class StopReason(Enum):
    """Why ``Executor.run(plan)`` stopped."""

    FEEDBACK_POINT = "feedback_point"
    PLAN_DONE = "plan_done"
    STEP_BUDGET = "step_budget"
    EXECUTION_ERROR = "execution_error"


@dataclass
class ExecutionSum:
    """Payload returned when plan execution pauses or finishes."""

    reason: StopReason
    steps: int = 0
    feedback_description: str | None = None
    feedback_prompt: str | None = None
    return_value: object | None = None
    error: str | None = None


class ReturnSignal(Exception):
    """Raised by a return node to unwind the call stack and deliver a value."""

    def __init__(self, value: object | None = None) -> None:
        super().__init__()
        self.value = value


class StepBudgetExceeded(Exception):
    """Raised when the episode step budget is exhausted."""


class FeedbackPause(Exception):
    """Internal: execution paused at a feedback node for runner/planner handoff."""

    def __init__(self, *, description: str, prompt: str) -> None:
        super().__init__(description)
        self.description = description
        self.prompt = prompt


class Executor:
    """Executes a plan against the environment and VLA policy."""

    def __init__(self, registry: SkillRegistry, env: LiberoEnv) -> None:
        self._registry = registry
        self._env = env
        self._vla = VLAClient()

    def run(self, plan: plan_ast.Plan) -> ExecutionSum:
        """Execute plan from scratch until feedback, done, budget, or error."""
        ctx: dict[str, object] = {}
        steps_at_start = self._env.steps
        self._last_skill_result: str | None = None
        try:
            self._execute_stmt(plan.body, ctx)
            return ExecutionSum(
                reason=StopReason.PLAN_DONE,
                steps=self._env.steps - steps_at_start,
                return_value=None,
            )
        except ReturnSignal as sig:
            return ExecutionSum(
                reason=StopReason.PLAN_DONE,
                steps=self._env.steps - steps_at_start,
                return_value=sig.value,
            )
        except FeedbackPause as pause:
            return ExecutionSum(
                reason=StopReason.FEEDBACK_POINT,
                steps=self._env.steps - steps_at_start,
                feedback_description=pause.description,
                feedback_prompt=pause.prompt,
            )
        except StepBudgetExceeded:
            return ExecutionSum(
                reason=StopReason.STEP_BUDGET,
                steps=self._env.steps - steps_at_start,
            )
        except Exception as exc:
            logger.exception("Executor execution error")
            return ExecutionSum(
                reason=StopReason.EXECUTION_ERROR,
                steps=self._env.steps - steps_at_start,
                error=str(exc),
            )

    def _episode_budget_remaining(self) -> int:
        """Return how many env steps remain before the episode cap."""
        return self._env.max_steps - self._env.steps

    def _check_episode_budget(self) -> None:
        """Raise if the episode step budget is already exhausted."""
        if self._episode_budget_remaining() <= 0:
            raise StepBudgetExceeded()

    def _run_arm_skill(
        self,
        skill: str,
        object_name: str,
        location: str | None,
    ) -> dict[str, object]:
        """Run a low-level arm skill by querying VLA chunks until budget or cap."""
        cleaned = re.sub(r"_\d+$", "", object_name).replace("_", " ")
        if skill == "arm.grab":
            task = f"pick up the {cleaned}"
        elif skill == "arm.place":
            task = f"put the {cleaned} in the {location}"
        else:
            raise ValueError(f"unsupported arm skill: {skill!r}")

        task_success = self._env.success()
        if task_success:
            return {"status": "attempted", "env_steps": 0, "task_success": True}

        skill_steps = 0
        steps_at_start = self._env.steps

        while (
            not self._env.success()
            and skill_steps < config.MAX_SKILL_STEPS
            and self._episode_budget_remaining() > 0
        ):
            image, wrist, state = self._env.observe()
            chunk = self._vla.predict(image, wrist, state, task)
            for action in chunk[: config.REPLAN_STEPS]:
                if (
                    skill_steps >= config.MAX_SKILL_STEPS
                    or self._episode_budget_remaining() <= 0
                ):
                    break
                self._env.step(action)
                skill_steps += 1
                if self._env.success():
                    break

        env_steps = self._env.steps - steps_at_start
        return {
            "status": "attempted",
            "env_steps": env_steps,
            "task_success": self._env.success(),
        }

    def _execute_stmt(
        self, node: plan_ast.StmtNode, ctx: dict[str, object]
    ) -> object | None:
        """Execute one statement node (recursively for compound statements)."""
        if isinstance(node, plan_ast.SeqNode):
            last: object | None = None
            for step in node.steps:
                last = self._execute_stmt(step, ctx)
            return last

        if isinstance(node, plan_ast.IfNode):
            if self._evaluate_expr(node.condition, ctx):
                return self._execute_stmt(node.then_branch, ctx)
            if node.else_branch is not None:
                return self._execute_stmt(node.else_branch, ctx)
            return None

        if isinstance(node, plan_ast.WhileNode):
            iterations = 0
            while self._evaluate_expr(node.condition, ctx):
                if iterations >= node.max_iterations:
                    raise RuntimeError(
                        f"while loop exceeded max_iterations={node.max_iterations}"
                    )
                self._execute_stmt(node.body, ctx)
                iterations += 1
            return None

        if isinstance(node, plan_ast.CallNode):
            self._check_episode_budget()
            evaluated_args = {
                key: self._evaluate_expr(value, ctx) for key, value in node.args.items()
            }
            args_str = ", ".join(f"{k}={v!r}" for k, v in evaluated_args.items())
            logger.info("Uses skill: %s(%s)", node.skill, args_str)

            if node.skill in _ARM_SKILLS:
                object_name = str(evaluated_args.get("object_name", ""))
                location = evaluated_args.get("location")
                location_str = str(location) if location is not None else None
                result = self._run_arm_skill(node.skill, object_name, location_str)
            else:
                skill = self._registry.get(node.skill)
                result = skill(**evaluated_args)

            self._last_skill_result = str(result)
            logger.info("Skill result (%s): %r", node.skill, result)
            if node.assign_to is not None:
                ctx[node.assign_to] = result
            return result

        if isinstance(node, plan_ast.FeedbackNode):
            prompt = node.description
            if self._last_skill_result:
                prompt = (
                    f"{prompt}\n\nPrior skill result: {self._last_skill_result}"
                )
            logger.info("Feedback checkpoint: %r", node.description)
            raise FeedbackPause(description=node.description, prompt=prompt)

        if isinstance(node, plan_ast.ReturnNode):
            value = (
                self._evaluate_expr(node.value, ctx) if node.value is not None else None
            )
            raise ReturnSignal(value=value)

        raise TypeError(f"Unsupported statement node: {type(node)!r}")

    def _evaluate_expr(self, expr: plan_ast.Expr, ctx: dict[str, object]) -> object:
        """Evaluate one expression node against the current variable bindings."""
        if isinstance(expr, plan_ast.LiteralExpr):
            return expr.value

        if isinstance(expr, plan_ast.VarExpr):
            if expr.name not in ctx:
                raise NameError(f"Variable {expr.name!r} is not bound")
            return ctx[expr.name]

        if isinstance(expr, plan_ast.BinOpExpr):
            if expr.op == "and":
                left = self._evaluate_expr(expr.left, ctx)
                if not left:
                    return left
                return self._evaluate_expr(expr.right, ctx)

            if expr.op == "or":
                left = self._evaluate_expr(expr.left, ctx)
                if left:
                    return left
                return self._evaluate_expr(expr.right, ctx)

            if expr.op == "in":
                left = self._evaluate_expr(expr.left, ctx)
                right = self._evaluate_expr(expr.right, ctx)
                if right is None:
                    return False
                return left in right

            left = self._evaluate_expr(expr.left, ctx)
            right = self._evaluate_expr(expr.right, ctx)

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
                raise TypeError(
                    f"Incomparable operands for {expr.op}: {left!r}, {right!r}"
                ) from exc

            raise ValueError(f"Unsupported binary operator: {expr.op!r}")

        if isinstance(expr, plan_ast.NotExpr):
            return not self._evaluate_expr(expr.operand, ctx)

        raise TypeError(f"Unsupported expression node: {type(expr)!r}")
