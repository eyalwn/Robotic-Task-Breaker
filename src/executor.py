"""Execute linear skill step lists against LIBERO and return summaries."""
from __future__ import annotations

import logging

import config
from libero_env import LiberoEnv
from plan import Plan
from vla_client import VLAClient

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
        error: str | None = None,
    ) -> None:
        self.reason = reason
        self.steps = steps
        self.error = error


class Executor:
    """Executes a linear step list against the environment and VLA policy."""

    def __init__(self, env: LiberoEnv) -> None:
        self._env = env
        self._vla = VLAClient()

    def run(self, steps: Plan) -> ExecutionSum:
        """Execute steps in order until done, subtask timeout, or error."""
        steps_at_start = self._env.steps
        try:
            for step in steps:
                logger.info("skill: %s", step["skill"])
                if not self._run_skill(step):
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

    def _task_string(self, step: dict[str, str]) -> str:
        """Build the VLA task string for one step."""
        skill = step["skill"]
        if skill == "grab":
            return f"pick up the {step['object_name']}"
        if skill == "place":
            position = step["position"]
            return (
                f"place the {step['object_name']} {position} the {step['target_name']}"
            )
        if skill == "close":
            return f"close the {step['target_name']}"
        if skill == "turn_on":
            return f"turn on the {step['target_name']}"
        return skill

    def _run_skill(self, step: dict[str, str]) -> bool:
        """Run one skill until subtask success or per-skill cap. Returns subtask ok."""
        prompt = self._task_string(step)
        skill_steps = 0
        while skill_steps < config.MAX_SKILL_STEPS:
            image, wrist, state = self._env.observe()
            actions_chunk = self._vla.predict(image, wrist, state, prompt)
            for action in actions_chunk:
                if skill_steps >= config.MAX_SKILL_STEPS:
                    break
                self._env.step(action)
                skill_steps += 1
            if self._env.check_subtask(step):
                return True
        return self._env.check_subtask(step)
