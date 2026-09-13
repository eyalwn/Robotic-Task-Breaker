"""Run a bare VLA baseline on VLABench (no planner or executor).

Loops ``BASELINE_EPISODES`` full-horizon episodes with native
``env.task_description`` and prints aggregate success rate to stdout.
"""
from __future__ import annotations

import argparse
import logging
import sys

import config
from envs.vlabench_env import VLABenchEnv, get_evaluation_episode_configs
from utils import configure_logging
from vla_client import VLAClient

logger = logging.getLogger(__name__)


def run_episode(env, vla: VLAClient) -> bool:
    """Run one full-horizon episode. Return True when the task succeeds."""
    while env.steps < config.MAX_EPISODE_STEPS:
        if env.success():
            return True
        images, state = env.observe()
        actions = vla.predict(images, state, env.task_description)
        for action in actions:
            if env.steps >= config.MAX_EPISODE_STEPS:
                return env.success()
            env.step(action)
            if env.success():
                return True
    return env.success()


def load_evaluation_episode_configs(task: str, episodes: int) -> list[dict[str, object]]:
    """Load the first N evaluation configs for one task."""
    configs = get_evaluation_episode_configs(task)
    if episodes > len(configs):
        raise RuntimeError(
            f"requested {episodes} episodes, but evaluation has {len(configs)} "
            f"for {task!r}"
        )
    return configs[:episodes]


def run_baseline(task: str, episodes: int) -> tuple[int, int]:
    """Run ``episodes`` independent raw-policy episodes. Return (successes, total)."""
    vla = VLAClient()
    successes = 0
    episode_configs = load_evaluation_episode_configs(task, episodes)
    for episode_idx, episode_config in enumerate(episode_configs):
        env = VLABenchEnv.from_track_episode(task, episode_config)
        try:
            logger.info(
                "baseline episode %s/%s task=%r instruction=%r",
                episode_idx + 1,
                episodes,
                task,
                env.task_description,
            )
            if run_episode(env, vla):
                successes += 1
                logger.info("episode %s succeeded", episode_idx + 1)
            else:
                logger.info("episode %s failed (step cap or goal not met)", episode_idx + 1)
        finally:
            env.close()
    return successes, len(episode_configs)


def main() -> int:
    """Parse CLI args, run the baseline campaign, and return an exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="select_fruit")
    parser.add_argument("--episodes", type=int, default=config.BASELINE_EPISODES)
    args = parser.parse_args()
    configure_logging()

    if args.episodes < 1:
        print(f"FAIL task={args.task!r} error='episodes must be >= 1'")
        return 1

    try:
        successes, total = run_baseline(args.task, args.episodes)
    except Exception as exc:
        logger.exception("baseline campaign failed")
        print(f"FAIL task={args.task!r} error={exc!r}")
        return 1

    rate = successes / total
    print(
        f"PASS task={args.task!r} baseline successes={successes}/{total} "
        f"rate={rate:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
