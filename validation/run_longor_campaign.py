"""Run Longor on the fixed M3 cook_dishes evaluation episodes."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from envs.vlabench_env import validate_evaluation_episode_range

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE = _REPO_ROOT / "examples" / "run_longor_task.py"
_TASK = "cook_dishes"


def main() -> int:
    """Run ``episodes`` consecutive M3 Longor episodes and return an exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int)
    args = parser.parse_args()

    if args.episodes < 1:
        print(f"FAIL task={_TASK!r} error='episodes must be >= 1'")
        return 1

    try:
        validate_evaluation_episode_range(_TASK, 0, args.episodes)
    except Exception as exc:
        print(f"FAIL task={_TASK!r} task_idx=0 error={exc!r}")
        return 1

    successes = 0
    for task_idx in range(args.episodes):
        result = subprocess.run(
            [
                sys.executable,
                str(_EXAMPLE),
                "--task",
                _TASK,
                "--task-idx",
                str(task_idx),
            ],
            cwd=_REPO_ROOT,
            check=False,
        )
        if result.returncode == 0:
            successes += 1

    total = args.episodes
    rate = successes / total
    print(
        f"PASS task={_TASK!r} task_idx=0 episodes={total} "
        f"successes={successes}/{total} rate={rate:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
