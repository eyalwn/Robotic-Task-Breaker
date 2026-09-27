"""Execute linear skill step lists and return summaries."""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import config
from plan import Plan
from vla_client import VLAClient

if TYPE_CHECKING:
    from envs.vlabench_env import VLABenchEnv

logger = logging.getLogger(__name__)

_Tracker = tuple[bool, float, int]


class StopReason:
    """Why ``Executor.run(steps)`` stopped."""

    PLAN_DONE = "plan_done"
    SUBTASK_TIMEOUT = "subtask_timeout"
    EPISODE_TIMEOUT = "episode_timeout"
    SKILL_ABORT = "skill_abort"
    EXECUTION_ERROR = "execution_error"


class SkillTimeout(Exception):
    """Per-skill step cap reached without subtask success."""


class EpisodeTimeout(Exception):
    """episode step cap reached without subtask success."""


class SkillAbort(Exception):
    """Skill ended early without subtask success (stall or drop)."""


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


def _advance_progress(
    best: float,
    stall: int,
    current: float,
    why: str,
) -> tuple[float, int]:
    """Update running-min patience; raise SkillAbort when progress stalls."""
    if current < best - config.TRANSFER_PROGRESS_EPS:
        return current, 0
    stall += 1
    if stall >= config.SKILL_STALL_STEPS:
        raise SkillAbort(f"transfer stalled {why}")
    return best, stall


class Executor:
    """Executes a linear step list against the environment and VLA policy."""

    def __init__(self, env: VLABenchEnv) -> None:
        self._env = env
        self._vla = VLAClient()

    def run(self, steps: Plan) -> ExecutionSum:
        """Execute steps in order until done, timeout, abort, or error."""
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
        except SkillAbort as exc:
            logger.info("skill aborted: %s", exc)
            return ExecutionSum(
                reason=StopReason.SKILL_ABORT,
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
        if skill == "open_drawer":
            return "Open the drawer"
        if skill == "transfer":
            object_name = step["object_name"]
            target_name = step["target_name"]
            return f"Put the {object_name} into the {target_name}"
        raise KeyError(f"unsupported skill: {skill!r}")

    def _run_step(self, step: dict[str, str]) -> None:
        """Run one skill until subtask success.

        Raise SkillTimeout, EpisodeTimeout, or SkillAbort.
        """
        prompt = self._build_vla_task(step)
        logger.debug("full skill prompt: %s", prompt)
        skill_steps = 0
        tracker: _Tracker = (False, math.inf, 0)
        while True:
            images, state = self._env.observe()
            actions_chunk = self._vla.predict(images, state, prompt)
            for action in actions_chunk:
                is_done = self._env.step(action)
                skill_steps += 1
                is_done = is_done or self._env.check_subtask(step)
                if is_done:
                    return
                self._enforce_max_steps_budget(step, skill_steps)
                if step["skill"] == "transfer":
                    distances = self._env.transfer_distances(step)
                    logger.debug("steps=%d, distances=%s, check_subtask=%s",
                        skill_steps,distances,is_done)
                    tracker = self._abort_if_stuck(step, tracker, distances)
    
    def _enforce_max_steps_budget(self, step, skill_steps):
        """Raise SkillTimeout / EpisodeTimeout if either MAX_SKILL_STEPS / 
            MAX_EPISODE_STEPS was exceeded."""
        max_skill_steps = config.MAX_SKILL_STEPS[step["skill"]]
        if skill_steps >= max_skill_steps:
            raise SkillTimeout(
                f"{step['skill']} exceeded {max_skill_steps} steps")
        elif self._env.steps >= config.MAX_EPISODE_STEPS:
            raise EpisodeTimeout(
                f"episode hits maximum steps-budget: {self._env.steps} steps")
    
    def _abort_if_stuck(
        self,
        step: dict[str, str],
        tracker: _Tracker,
        distances: tuple[float, float],
    ) -> _Tracker:
        """Raise SkillAbort if transfer GT progress stalled or the object dropped."""
        if step["skill"] != "transfer":
            return tracker
        eef_obj, obj_tgt = distances
        if not math.isfinite(eef_obj) or not math.isfinite(obj_tgt):
            raise RuntimeError("non-finite transfer distances")
        held, best, stall = tracker
        if not held:
            if eef_obj <= config.TRANSFER_HOLD_DISTANCE:
                return True, obj_tgt, 0
            best, stall = _advance_progress(
                best, stall, eef_obj, "approaching object"
            )
            return False, best, stall
        if eef_obj > config.TRANSFER_DROP_DISTANCE:
            raise SkillAbort("transfer dropped object")
        best, stall = _advance_progress(
            best, stall, obj_tgt, "moving to target"
        )
        return True, best, stall
