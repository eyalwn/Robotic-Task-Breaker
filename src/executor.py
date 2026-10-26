"""Execute a linear plan: turn each skill into a VLA prompt and step the env.

Each skill runs until the env's subtask check passes, or until a cap or error.
"""
from __future__ import annotations

import logging
from typing import Any

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
    """Explicit per-skill step cap reached without subtask success."""


class EpisodeTimeout(Exception):
    """Episode step cap reached without subtask success."""

    def __init__(self, message: str, *,
                 partial_telemetry: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.partial_telemetry = partial_telemetry


class ExecutionSum:
    """Payload returned when plan execution pauses or finishes."""

    def __init__(self, *, reason: str, steps: int = 0, error: str = "",
                 next_step_idx: int = 0,
                 skill_telemetry: list[dict[str, Any]] | None = None) -> None:
        """Capture why execution stopped, resume index, and per-skill telemetry."""
        self.reason = reason
        self.steps = steps
        self.error = error
        self.next_step_idx = next_step_idx
        self.skill_telemetry = list(skill_telemetry or [])


class Executor:
    """Executes a linear step list against the environment and VLA policy."""

    def __init__(self, env: SimEnv) -> None:
        """Bind the environment and a VLA client."""
        self._env = env
        self._vla = VLAClient()

    def run(self, steps: Plan,
            skill_step_limits: dict[str, int] | None = None) -> ExecutionSum:
        """Execute steps until done, explicit skill cap, horizon, or error."""
        limits = _validate_skill_step_limits(skill_step_limits)
        steps_at_start = self._env.steps
        telemetry: list[dict[str, Any]] = []
        for idx, step in enumerate(steps):
            try:
                logger.info("executing step: %s", step["skill"])
                telemetry.append(self._run_step(step, limits))
                logger.info("skill done successfully: %s", step["skill"])
            except SkillTimeout as exc:
                logger.info("skill timed out: %s", exc)
                return ExecutionSum(reason=StopReason.SUBTASK_TIMEOUT,
                                    steps=self._env.steps - steps_at_start,
                                    error=str(exc), next_step_idx=idx,
                                    skill_telemetry=telemetry)
            except EpisodeTimeout as exc:
                logger.info("episode timed out: %s", exc)
                if exc.partial_telemetry is not None:
                    telemetry.append(exc.partial_telemetry)
                return ExecutionSum(reason=StopReason.EPISODE_TIMEOUT,
                                    steps=self._env.steps - steps_at_start,
                                    error=str(exc), next_step_idx=idx,
                                    skill_telemetry=telemetry)
            except Exception as exc:
                logger.exception("executor error")
                return ExecutionSum(reason=StopReason.EXECUTION_ERROR,
                                    steps=self._env.steps - steps_at_start,
                                    error=str(exc), next_step_idx=idx,
                                    skill_telemetry=telemetry)
        return ExecutionSum(reason=StopReason.PLAN_DONE,
                            steps=self._env.steps - steps_at_start,
                            next_step_idx=len(steps), skill_telemetry=telemetry)

    def _build_vla_task(self, step: dict[str, str]) -> str:
        """Build the VLA task string for one step."""
        skill = step["skill"]
        if skill == "open":
            return f"open the {step['object_name']}"
        if skill == "close":
            return f"close the {step['object_name']}"
        raise KeyError(f"unsupported skill: {skill!r}")

    def _run_step(self, step: dict[str, str],
                  skill_step_limits: dict[str, int] | None) -> dict[str, Any]:
        """Run one skill until subtask success or a hard cap.

        Raise SkillTimeout or EpisodeTimeout.
        """
        skill_key = step["skill"]
        if self._env.steps >= self._env.horizon:
            raise EpisodeTimeout(
                f"episode hits maximum steps-budget: {self._env.steps} steps",
                partial_telemetry={"skill": skill_key, "steps": 0,
                                   "threshold_crossed": False})
        prompt = self._build_vla_task(step)
        logger.debug("full skill prompt: %s", prompt)
        threshold = config.MAX_SKILL_STEPS[skill_key]
        skill_steps = 0
        threshold_crossed = False
        while True:
            images, state = self._env.observe()
            actions_chunk = self._vla.predict(images, state, prompt)
            for action in actions_chunk:
                is_done = self._env.step(action)
                skill_steps += 1
                if skill_steps >= threshold:
                    threshold_crossed = True
                is_done = is_done or self._env.check_subtask(step)
                logger.debug("steps=%d, check_subtask=%s", skill_steps, is_done)
                if is_done:
                    return {"skill": skill_key, "steps": skill_steps,
                            "threshold_crossed": threshold_crossed}
                self._enforce_hard_caps(step, skill_steps, skill_step_limits,
                                        threshold_crossed)

    def _enforce_hard_caps(self, step: dict[str, str], skill_steps: int,
                           skill_step_limits: dict[str, int] | None,
                           threshold_crossed: bool) -> None:
        """Raise SkillTimeout / EpisodeTimeout when an explicit cap is exceeded."""
        skill_key = step["skill"]
        if skill_step_limits is not None:
            limit = skill_step_limits.get(skill_key)
            if limit is not None and skill_steps >= limit:
                raise SkillTimeout(
                    f"{step['skill']} exceeded explicit cap of {limit} steps")
        if self._env.steps >= self._env.horizon:
            raise EpisodeTimeout(
                f"episode hits maximum steps-budget: {self._env.steps} steps",
                partial_telemetry={"skill": skill_key, "steps": skill_steps,
                                   "threshold_crossed": threshold_crossed})


def _validate_skill_step_limits(
        skill_step_limits: dict[str, int] | None) -> dict[str, int] | None:
    """Validate optional per-skill hard caps; fail fast on bad input."""
    if skill_step_limits is None:
        return None
    validated: dict[str, int] = {}
    for skill, cap in skill_step_limits.items():
        if skill not in config.MAX_SKILL_STEPS:
            raise ValueError(
                f"unsupported skill in skill_step_limits: {skill!r}")
        if cap < 1:
            raise ValueError(
                f"skill_step_limits[{skill!r}] must be >= 1, got {cap}")
        validated[skill] = cap
    return validated
