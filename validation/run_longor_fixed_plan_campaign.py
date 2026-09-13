"""Run planner-bypass Longor campaigns with a fixed transfer target token."""
from __future__ import annotations

import argparse
import re

import config
from envs.sim_env import SimEnv
from envs.vlabench_env import validate_evaluation_episode_range
from executor import Executor, StopReason
from utils import configure_logging
from vla_client import VLAClient

configure_logging()

_GOAL_PATTERN = re.compile(r"^Put the (?P<object>.+) into the (?P<target>.+)$")
_ORACLE_TARGET_NAME = "plate_seen"


class _FixedPromptVLAClient(VLAClient):
    """Send a fixed task prompt while the executor retains oracle step names."""

    def __init__(self, task_prompt: str) -> None:
        super().__init__()
        self._task_prompt = task_prompt

    def predict(
        self,
        images: tuple[bytes, ...],
        state: list[float],
        task: str,
    ) -> list[list[float]]:
        """Request the policy action chunk using the fixed validation prompt."""
        return super().predict(images, state, self._task_prompt)


def _extract_transfer_object(task_description: str) -> str:
    """Extract the object token from a VLABench transfer goal string."""
    match = _GOAL_PATTERN.match(task_description.strip())
    if match is None:
        raise RuntimeError(
            "unsupported task instruction format for planner-bypass transfer: "
            f"{task_description!r}"
        )
    return match.group("object").strip()


def _run_fixed_transfer_episode(
    task: str,
    task_idx: int,
    prompt_target: str,
    max_replans: int,
) -> tuple[bool, int, str]:
    """Run one episode with a single fixed transfer step and return outcome."""
    env = SimEnv(task, task_idx)
    executor = Executor(env=env)
    total_steps = 0
    error = ""
    success = False
    status = "execution_error"
    try:
        object_name = _extract_transfer_object(env.task_description)
        task_prompt = f"Put the {object_name} into the {prompt_target}"
        step = {
            "skill": "transfer",
            "object_name": object_name,
            "target_name": _ORACLE_TARGET_NAME,
        }
        executor._vla = _FixedPromptVLAClient(task_prompt)
        print(
            f"INFO task={task!r} task_idx={task_idx} "
            f"vla_prompt={task_prompt!r} oracle_target={_ORACLE_TARGET_NAME!r}"
        )
        for attempt in range(max_replans + 1):
            execution_sum = executor.run([step])
            total_steps += execution_sum.steps
            if execution_sum.reason == StopReason.PLAN_DONE:
                success = bool(env.success())
                status = "done"
                break
            status = "execution_error"
            if execution_sum.reason == StopReason.SUBTASK_TIMEOUT:
                error = execution_sum.error or "subtask step cap exceeded"
            elif execution_sum.reason == StopReason.SKILL_ABORT:
                error = execution_sum.error or "skill aborted"
            else:
                error = execution_sum.error or ""
                break
            if attempt < max_replans:
                print(f"INFO task={task!r} task_idx={task_idx} replanning_after={execution_sum.reason}")
    finally:
        env.close()
    return success, total_steps, (error if error else status)


def main() -> int:
    """Run one fixed-target campaign over consecutive episodes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="select_fruit")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--prompt-target", required=True)
    parser.add_argument("--max-replans", type=int, default=config.MAX_REPLANS)
    args = parser.parse_args()

    if args.episodes < 1:
        print(f"FAIL task={args.task!r} error='episodes must be >= 1'")
        return 1
    if args.start_idx < 0:
        print(f"FAIL task={args.task!r} error='start_idx must be >= 0'")
        return 1
    if args.max_replans < 0:
        print(f"FAIL task={args.task!r} error='max_replans must be >= 0'")
        return 1

    try:
        validate_evaluation_episode_range(args.task, args.start_idx, args.episodes)
    except Exception as exc:
        print(f"FAIL task={args.task!r} task_idx={args.start_idx} error={exc!r}")
        return 1

    successes = 0
    for task_idx in range(args.start_idx, args.start_idx + args.episodes):
        try:
            success, steps, detail = _run_fixed_transfer_episode(
                task=args.task,
                task_idx=task_idx,
                prompt_target=args.prompt_target,
                max_replans=args.max_replans,
            )
        except Exception as exc:
            print(f"FAIL task={args.task!r} task_idx={task_idx} success=False error={exc!r}")
            continue
        verdict = "PASS" if success else "FAIL"
        print(
            f"{verdict} task={args.task!r} task_idx={task_idx} "
            f"success={success} prompt_target={args.prompt_target!r} "
            f"steps={steps} detail={detail}"
        )
        if success:
            successes += 1

    rate = successes / args.episodes
    print(
        f"PASS task={args.task!r} start_idx={args.start_idx} episodes={args.episodes} "
        f"prompt_target={args.prompt_target!r} "
        f"successes={successes}/{args.episodes} rate={rate:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
