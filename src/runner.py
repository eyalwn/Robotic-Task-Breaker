"""Wire and run planner-executor for one task."""

import logging
from typing import Literal

import config
from envs.sim_env import SimEnv
from executor import Executor, StopReason
from plan import Plan, get_stop_reason, is_stop_plan
from planner import LLMPlanner
from skills import skill_catalog

logger = logging.getLogger(__name__)

RunStatus = Literal["done", "plan_failed", "stopped", "execution_error"]


class Runner:
    """Orchestrates observe → plan → execute with suffix-preserving retries."""

    def __init__(self, task: str, task_idx: int = 0) -> None:
        """Wire env, planner, and executor for one task."""
        self._env = SimEnv(task, task_idx)
        self._goal = self._env.task_description
        try:
            self._planner = LLMPlanner(self._goal, skill_catalog())
            self._executor = Executor(env=self._env)
        except Exception:
            self._env.close()
            raise

    def run(self) -> bool:
        """Run the task loop and return the final success verdict."""
        status: RunStatus = "plan_failed"
        error = ""
        stop_reason = ""
        total_steps = 0
        try:
            images, _state = self._env.observe()
            current_plan = self._planner.plan(images[0])
            if is_stop_plan(current_plan):
                status = "stopped"
                error = get_stop_reason(current_plan)
                stop_reason = "planner_stop"
            else:
                remaining_steps: Plan = list(current_plan)
                retry_count = 0
                while remaining_steps:
                    execution_sum = self._executor.run(remaining_steps)
                    stop_reason = execution_sum.reason
                    total_steps += execution_sum.steps
                    if execution_sum.reason == StopReason.PLAN_DONE:
                        status = "done"
                        break
                    status = "execution_error"
                    if execution_sum.reason == StopReason.SUBTASK_TIMEOUT:
                        error = execution_sum.error or "subtask step cap exceeded"
                        remaining_steps = remaining_steps[execution_sum.next_step_idx :]
                        if retry_count < config.MAX_REPLANS and remaining_steps:
                            retry_count += 1
                            logger.info(
                                "retrying unfinished plan suffix after %s",
                                execution_sum.reason,
                            )
                            continue
                        break
                    if execution_sum.reason == StopReason.EPISODE_TIMEOUT:
                        error = execution_sum.error or "episode step cap exceeded"
                        break
                    if execution_sum.reason == StopReason.EXECUTION_ERROR:
                        error = execution_sum.error or ""
                        break
                    error = f"unexpected stop reason: {execution_sum.reason}"
                    break
        except RuntimeError as exc:
            error = str(exc)
        finally:
            success = self._verdict()
            self._log_status(status, success, error, stop_reason, total_steps)
            self._env.close()
        return success

    def _verdict(self) -> bool:
        """Compute final success verdict from the terminal simulator state."""
        return bool(self._env.success())

    def _log_status(
        self,
        status: RunStatus,
        success: bool,
        error: str,
        stop_reason: str,
        total_steps: int,
    ) -> None:
        """Emit outcome-specific logs and a compact run summary."""
        if success:
            logger.info(
                "Task succeeded: %r (status=%s, stop_reason=%s)",
                self._goal,
                status,
                stop_reason or "none",
            )
        elif status == "done":
            logger.info(
                "Task not achieved: %r (plan finished; env goal check failed)",
                self._goal,
            )
        elif status == "plan_failed":
            logger.info("Task failed: %r (planner error)", self._goal)
        elif status == "stopped":
            logger.info("Task stopped: %r (%s)", self._goal, error)
        else:
            logger.info("Task failed: %r (status=%s)", self._goal, status)

        summary_parts = [
            f"success={success}",
            f"plan_status={status}",
            f"steps={total_steps}",
        ]
        if stop_reason:
            summary_parts.append(f"stop_reason={stop_reason}")
        if error:
            summary_parts.append(f"error={error}")

        logger.info("Run summary: %s", ", ".join(summary_parts))
