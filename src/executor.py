"""Execute linear skill step lists and return summaries."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import config
from plan import Plan
from vla_client import VLAClient

if TYPE_CHECKING:
    from envs.vlabench_env import VLABenchEnv

logger = logging.getLogger(__name__)


class StopReason:
    """Why ``Executor.run(steps)`` stopped."""

    PLAN_DONE = "plan_done"
    SUBTASK_TIMEOUT = "subtask_timeout"
    EXECUTION_ERROR = "execution_error"


class ExecutionSum:
    """Payload returned when plan execution pauses or finishes."""

    def __init__(
        self,
        *,
        reason: str,
        steps: int = 0,
        error: str = "",
    ) -> None:
        self.reason = reason
        self.steps = steps
        self.error = error


class Executor:
    """Executes a linear step list against the environment and VLA policy."""

    def __init__(self, env: VLABenchEnv) -> None:
        self._env = env
        self._vla = VLAClient()

    def run(self, steps: Plan) -> ExecutionSum:
        """Execute steps in order until done, subtask timeout, or error."""
        steps_at_start = self._env.steps
        try:
            for step in steps:
                logger.info("executing step: %s", step["skill"])
                if self._run_step_successfuly(step):
                    logger.info("skill done successfuly: %s", step["skill"])
                else:
                    return ExecutionSum(
                        reason=StopReason.SUBTASK_TIMEOUT,
                        steps=self._env.steps - steps_at_start,
                    )
            return ExecutionSum(
                reason=StopReason.PLAN_DONE,
                steps=self._env.steps - steps_at_start,
            )
        except Exception as exc:
            logger.exception("executor error")
            return ExecutionSum(
                reason=StopReason.EXECUTION_ERROR,
                steps=self._env.steps - steps_at_start,
                error=str(exc),
            )

    def _build_vla_task(self, step: dict[str, str]) -> str:
        """Build the VLA task string for one step."""
        skill = step["skill"]
        if skill == "transfer":
            return f"Put the {step['object_name']} into the {step['target_name']}"
        raise KeyError(f"unsupported skill: {skill!r}")

    def _run_step_successfuly(self, step: dict[str, str]) -> bool:
        """Run one skill until subtask success or per-skill cap. Returns subtask ok."""
        prompt = self._build_vla_task(step)
        max_skill_steps = config.MAX_SKILL_STEPS[step["skill"]]
        logger.debug("full skill prompt: %s", prompt)
        skill_steps, is_done = 0, False
        while skill_steps < max_skill_steps and not is_done:
            images, state = self._env.observe()
            actions_chunk = self._vla.predict(images, state, prompt)
            for action in actions_chunk:
                is_done = self._env.step(action)
                skill_steps += 1
                if skill_steps >= max_skill_steps or is_done:
                    break
            is_done = is_done or self._env.check_subtask(step)
            logger.debug("steps=%d, eef_delta=%s, check_subtask=%s",
                skill_steps, self._env.eef_goal_distance(step), is_done)
        return is_done
