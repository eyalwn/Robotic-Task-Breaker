"""Run Longor on consecutive evaluation episodes for one task."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from envs.vlabench_env import validate_evaluation_episode_range

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE = _REPO_ROOT / "examples" / "run_longor_task.py"


def main() -> int:
    """Run consecutive Longor episodes for one task and return an exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="select_fruit")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    task = args.task
    start_idx = args.start_idx
    episodes = args.episodes

    if episodes < 1:
        print(f"FAIL task={task!r} error='episodes must be >= 1'")
        return 1
    if start_idx < 0:
        print(f"FAIL task={task!r} error='start_idx must be >= 0'")
        return 1

    try:
        validate_evaluation_episode_range(task, start_idx, episodes)
    except Exception as exc:
        print(f"FAIL task={task!r} task_idx={start_idx} error={exc!r}")
        return 1

    successes = 0
    for task_idx in range(start_idx, start_idx + episodes):
        result = subprocess.run(
            [
                sys.executable,
                str(_EXAMPLE),
                "--task",
                task,
                "--task-idx",
                str(task_idx),
            ],
            cwd=_REPO_ROOT,
            check=False,
        )
        if result.returncode == 0:
            successes += 1

    rate = successes / episodes
    print(
        f"PASS task={task!r} start_idx={start_idx} episodes={episodes} "
        f"successes={successes}/{episodes} rate={rate:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
