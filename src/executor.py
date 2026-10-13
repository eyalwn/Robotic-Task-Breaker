"""Execute linear skill step lists and return summaries."""
from __future__ import annotations

import logging

import config
from envs.sim_env import SimEnv
from plan import Plan
from vla_client import VLAClient

logger = logging.getLogger(__name__)


class StopReason:
    """Why ``Executor.run(steps)`` stopped."""

    PLAN_DONE = "plan_done"
    SUBTASK_TIMEOUT = "subtask_timeout"
    EPISODE_TIMEOUT = "episode_timeout"
    EXECUTION_ERROR = "execution_error"


class SkillTimeout(Exception):
    """Per-skill step cap reached without subtask success."""


class EpisodeTimeout(Exception):
    """episode step cap reached without subtask success."""


class ExecutionSum:
    """Payload returned when plan execution pauses or finishes."""

    def __init__(self, *, reason: str, steps: int = 0, error: str = "") -> None:
        self.reason = reason
        self.steps = steps
        self.error = error


class Executor:
    """Executes a linear step list against the environment and VLA policy."""

    def __init__(self, env: SimEnv) -> None:
        self._env = env
        self._vla = VLAClient()

    def run(self, steps: Plan) -> ExecutionSum:
        """Execute steps in order until done, timeout, or error."""
        steps_at_start = self._env.steps
        try:
            for step in steps:
                logger.info("executing step: %s", step["skill"])
                self._run_step(step)
                logger.info("skill done successfully: %s", step["skill"])
            return ExecutionSum(
                reason=StopReason.PLAN_DONE,
                steps=self._env.steps - steps_at_start,
            )
        except SkillTimeout as exc:
            logger.info("skill timed out: %s", exc)
            return ExecutionSum(
                reason=StopReason.SUBTASK_TIMEOUT,
                steps=self._env.steps - steps_at_start,
                error=str(exc),
            )
        except EpisodeTimeout as exc:
            logger.info("episode timed out: %s", exc)
            return ExecutionSum(
                reason=StopReason.EPISODE_TIMEOUT,
                steps=self._env.steps - steps_at_start,
                error=str(exc),
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
        if skill == "open":
            return f"open the {step['object_name']}"
        if skill == "place":
            obj = step["object_name"]
            src = step["source_name"]
            dest = step["destination_name"]
            return (
                f"pick the {obj} from the {src} and place it on the {dest}")
        if skill == "put_in":
            obj = step["object_name"]
            src = step["source_name"]
            container = step["container_name"]
            return (
                f"pick the {obj} from the {src} and place it in the {container}"
            )
        if skill == "close":
            return f"close the {step['object_name']}"
        if skill == "turn_on":
            return f"turn on the {step['target_name']}"
        raise KeyError(f"unsupported skill: {skill!r}")

    def _run_step(self, step: dict[str, str]) -> None:
        """Run one skill until subtask success.

        Raise SkillTimeout or EpisodeTimeout.
        """
        prompt = self._build_vla_task(step)
        logger.debug("full skill prompt: %s", prompt)
        skill_steps = 0
        while True:
            images, state = self._env.observe()
            actions_chunk = self._vla.predict(images, state, prompt)
            for action in actions_chunk:
                is_done = self._env.step(action)
                skill_steps += 1
                is_done = is_done or self._env.check_subtask(step)
                logger.debug(
                    "steps=%d, check_subtask=%s",
                    skill_steps,
                    is_done,
                )
                if is_done:
                    return
                self._enforce_max_steps_budget(step, skill_steps)

    def _enforce_max_steps_budget(self, step, skill_steps):
        """Raise SkillTimeout / EpisodeTimeout if skill or episode cap exceeded."""
        skill_key = step["skill"]
        max_skill_steps = config.MAX_SKILL_STEPS[skill_key]
        if skill_steps >= max_skill_steps:
            raise SkillTimeout(
                f"{step['skill']} exceeded {max_skill_steps} steps")
        elif self._env.steps >= self._env.horizon:
            raise EpisodeTimeout(
                f"episode hits maximum steps-budget: {self._env.steps} steps")
