"""Run Longor on one or more VLABench evaluation episodes."""
from __future__ import annotations

import argparse
import sys

from envs.vlabench_env import validate_evaluation_episode_range
from runner import Runner
from utils import configure_logging


def run_campaign(task: str, task_idx: int, episodes: int) -> tuple[int, int]:
    """Run consecutive evaluation episodes. Return (successes, total)."""
    successes = 0
    for offset in range(episodes):
        idx = task_idx + offset
        if Runner(task, idx).run():
            successes += 1
    return successes, episodes


def main() -> int:
    """Parse CLI args, run Longor, and return an exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="select_fruit")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    configure_logging()

    if args.episodes < 1:
        print(
            f"FAIL task={args.task!r} task_idx={args.task_idx} "
            f"error='episodes must be >= 1'"
        )
        return 1

    try:
        if args.episodes == 1:
            success = Runner(args.task, args.task_idx).run()
        else:
            validate_evaluation_episode_range(
                args.task, args.task_idx, args.episodes
            )
            successes, total = run_campaign(args.task, args.task_idx, args.episodes)
            rate = successes / total
            print(
                f"PASS task={args.task!r} task_idx={args.task_idx} "
                f"episodes={total} successes={successes}/{total} rate={rate:.2f}"
            )
            return 0
    except Exception as exc:
        print(
            f"FAIL task={args.task!r} task_idx={args.task_idx} error={exc!r}"
        )
        return 1

    verdict = "PASS" if success else "FAIL"
    print(
        f"{verdict} task={args.task!r} task_idx={args.task_idx} success={success}"
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
