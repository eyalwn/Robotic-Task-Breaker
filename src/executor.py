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
_ARM_SKILLS = frozenset(
    {
        "arm.grab",
        "arm.place",
        "arm.transfer",
        "arm.open",
        "arm.close",
        "arm.push",
        "arm.pull",
    }
)


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

    def _clean_arm_name(self, name: str) -> str:
        """Normalize scene object names for VLA task strings."""
        return re.sub(r"_\d+$", "", name).replace("_", " ")

    def _match_conjunct_indices(
        self, primary_object: str, conjuncts: list[list[str]]
    ) -> list[int]:
        """Return conjunct indices matching a planner object name (exact, then containment)."""
        normalized = self._clean_arm_name(primary_object)
        candidates = [
            (index, self._clean_arm_name(str(conjunct[1])))
            for index, conjunct in enumerate(conjuncts)
            if len(conjunct) >= 2
        ]
        exact = [index for index, conjunct_name in candidates if conjunct_name == normalized]
        if exact:
            return exact
        return [
            index
            for index, conjunct_name in candidates
            if normalized in conjunct_name or conjunct_name in normalized
        ]

    def _require_arm_arg(
        self, args: dict[str, object], key: str, skill: str
    ) -> str:
        """Return a cleaned required arm-skill argument or raise."""
        value = args.get(key)
        if value is None or (isinstance(value, str) and not value):
            raise ValueError(f"{skill} requires {key!r}")
        return self._clean_arm_name(str(value))

    def _arm_primary_object_name(self, skill: str, args: dict[str, object]) -> str | None:
        """Return the raw primary object argument for subtask signal resolution."""
        if skill in ("arm.grab", "arm.place", "arm.transfer"):
            value = args.get("object_name")
        elif skill in ("arm.open", "arm.close", "arm.push", "arm.pull"):
            value = args.get("target_name")
        else:
            return None
        if value is None or (isinstance(value, str) and not value):
            return None
        return str(value)

    def _resolve_conjunct_index(self, primary_object: str) -> int | None:
        """Map a skill object name to exactly one goal conjunct, else budget-only."""
        goal_conjuncts = getattr(self._env, "goal_conjuncts", None)
        if goal_conjuncts is None:
            return None
        try:
            conjuncts = goal_conjuncts()
        except Exception:
            logger.warning(
                "goal_conjuncts() failed for %r; using step-budget-only termination",
                primary_object,
                exc_info=True,
            )
            return None

        matches = self._match_conjunct_indices(primary_object, conjuncts)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            logger.warning(
                "no goal conjunct matched %r; using step-budget-only termination",
                primary_object,
            )
        else:
            logger.warning(
                "ambiguous goal conjuncts %s for %r; using step-budget-only termination",
                matches,
                primary_object,
            )
        return None

    def _looks_like_grasp_object_id(
        self, name: str, conjuncts: list[list[str]]
    ) -> bool:
        """Return whether ``name`` is already a BDDL-style object instance token."""
        if re.search(r"_\d+$", name):
            return True
        return any(
            len(conjunct) >= 2 and str(conjunct[1]) == name for conjunct in conjuncts
        )

    def _resolve_grasp_object(self, primary_object: str) -> str | None:
        """Map a planner object name to a grasp-check object ID, else budget-only."""
        goal_conjuncts = getattr(self._env, "goal_conjuncts", None)
        conjuncts: list[list[str]] = []
        if goal_conjuncts is not None:
            try:
                conjuncts = goal_conjuncts()
            except Exception:
                logger.warning(
                    "goal_conjuncts() failed for %r; using step-budget-only termination",
                    primary_object,
                    exc_info=True,
                )

        matches = self._match_conjunct_indices(primary_object, conjuncts)
        if len(matches) == 1:
            return str(conjuncts[matches[0]][1])

        if self._looks_like_grasp_object_id(primary_object, conjuncts):
            return primary_object

        if not matches:
            logger.warning(
                "no goal conjunct matched %r; using step-budget-only termination",
                primary_object,
            )
        else:
            logger.warning(
                "ambiguous goal conjuncts %s for %r; using step-budget-only termination",
                matches,
                primary_object,
            )
        return None

    def _subtask_satisfied(
        self,
        skill: str,
        *,
        grasp_object: str | None,
        conjunct_index: int | None,
    ) -> bool:
        """Return whether the arm skill's subtask-completion signal is true."""
        if skill == "arm.grab":
            if grasp_object is None:
                return False
            is_grasping = getattr(self._env, "is_grasping", None)
            if is_grasping is None:
                return False
            try:
                return bool(is_grasping(grasp_object))
            except Exception:
                logger.warning(
                    "is_grasping(%r) failed; using step-budget-only termination",
                    grasp_object,
                    exc_info=True,
                )
                return False

        if conjunct_index is None:
            return False
        eval_conjunct = getattr(self._env, "eval_goal_conjunct", None)
        if eval_conjunct is None:
            return False
        try:
            return bool(eval_conjunct(conjunct_index))
        except Exception:
            logger.warning(
                "eval_goal_conjunct(%d) failed; using step-budget-only termination",
                conjunct_index,
                exc_info=True,
            )
            return False

    def _resolve_subtask_signal(
        self, skill: str, args: dict[str, object]
    ) -> tuple[str | None, int | None]:
        """Return grasp object name and/or goal-conjunct index for subtask checks."""
        primary_object = self._arm_primary_object_name(skill, args)
        if primary_object is None:
            return None, None
        if skill == "arm.grab":
            return self._resolve_grasp_object(primary_object), None
        return None, self._resolve_conjunct_index(primary_object)

    def _arm_skill_payload(
        self,
        *,
        env_steps: int,
        subtask_success: bool,
        task_success: bool,
    ) -> dict[str, object]:
        """Build the arm-skill return dict."""
        status = "done" if subtask_success or task_success else "timeout"
        return {
            "status": status,
            "env_steps": env_steps,
            "subtask_success": subtask_success,
            "task_success": task_success,
        }

    def _build_arm_task(self, skill: str, args: dict[str, object]) -> str:
        """Build the VLA task string for an executor-handled arm skill."""
        if skill == "arm.grab":
            obj = self._require_arm_arg(args, "object_name", skill)
            return f"pick up the {obj}"
        if skill == "arm.place":
            obj = self._require_arm_arg(args, "object_name", skill)
            loc = self._require_arm_arg(args, "location", skill)
            return f"put the {obj} in the {loc}"
        if skill == "arm.transfer":
            obj = self._require_arm_arg(args, "object_name", skill)
            loc = self._require_arm_arg(args, "location", skill)
            return f"move the {obj} to the {loc}"
        if skill == "arm.open":
            target = self._require_arm_arg(args, "target_name", skill)
            return f"open the {target}"
        if skill == "arm.close":
            target = self._require_arm_arg(args, "target_name", skill)
            return f"close the {target}"
        if skill == "arm.push":
            target = self._require_arm_arg(args, "target_name", skill)
            dest = self._require_arm_arg(args, "destination_or_direction", skill)
            return f"push the {target} to the {dest}"
        if skill == "arm.pull":
            target = self._require_arm_arg(args, "target_name", skill)
            dest = self._require_arm_arg(args, "destination_or_direction", skill)
            return f"pull the {target} to the {dest}"
        raise ValueError(f"unsupported arm skill: {skill!r}")

    def _run_arm_skill(
        self,
        skill: str,
        args: dict[str, object],
    ) -> dict[str, object]:
        """Run a low-level arm skill until subtask, task success, or budget."""
        task = self._build_arm_task(skill, args)
        grasp_object, conjunct_index = self._resolve_subtask_signal(skill, args)

        if self._env.success():
            return self._arm_skill_payload(
                env_steps=0,
                subtask_success=True,
                task_success=True,
            )

        skill_steps = 0
        steps_at_start = self._env.steps

        while skill_steps < config.MAX_SKILL_STEPS and self._episode_budget_remaining() > 0:
            if self._subtask_satisfied(
                skill,
                grasp_object=grasp_object,
                conjunct_index=conjunct_index,
            ) or self._env.success():
                break

            image, wrist, state = self._env.observe()
            chunk = self._vla.predict(image, wrist, state, task)
            for action in chunk[: config.VLA_ACTIONS_PER_CHUNK]:
                if (
                    skill_steps >= config.MAX_SKILL_STEPS
                    or self._episode_budget_remaining() <= 0
                ):
                    break
                self._env.step(action)
                skill_steps += 1
                if self._subtask_satisfied(
                    skill,
                    grasp_object=grasp_object,
                    conjunct_index=conjunct_index,
                ) or self._env.success():
                    break

        env_steps = self._env.steps - steps_at_start
        task_success = self._env.success()
        subtask_success = self._subtask_satisfied(
            skill,
            grasp_object=grasp_object,
            conjunct_index=conjunct_index,
        )
        return self._arm_skill_payload(
            env_steps=env_steps,
            subtask_success=subtask_success,
            task_success=task_success,
        )

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
                result = self._run_arm_skill(node.skill, evaluated_args)
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
