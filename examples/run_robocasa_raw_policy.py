"""Run bare Longor HTTP VLA on RoboCasa OpenDrawer (no planner or executor)."""
from __future__ import annotations

import argparse
import logging
import sys

import config
from envs.robocasa_env import RoboCasaEnv
from utils import configure_logging
from vla_client import VLAClient

logger = logging.getLogger(__name__)

_OPEN_DRAWER = "open_drawer"


def run_episode(env: RoboCasaEnv, vla: VLAClient, max_steps: int) -> bool:
    """Run one OpenDrawer episode through the Longor VLA HTTP client."""
    while env.steps < max_steps:
        if env.success():
            return True
        images, state = env.observe()
        actions = vla.predict(images, state, env.task_description)
        for action in actions:
            if env.steps >= max_steps:
                return env.success()
            env.step(action)
            if env.success():
                return True
    return env.success()


def main() -> int:
    """Run consecutive raw-policy OpenDrawer episodes and print aggregate rate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=config.BASELINE_EPISODES)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=config.MAX_SKILL_STEPS[_OPEN_DRAWER],
    )
    args = parser.parse_args()
    configure_logging()

    if args.episodes < 1:
        print(f"FAIL task={_OPEN_DRAWER!r} error='episodes must be >= 1'")
        return 1
    if args.max_steps < 1:
        print(f"FAIL task={_OPEN_DRAWER!r} error='max_steps must be >= 1'")
        return 1

    vla = VLAClient()
    successes = 0
    try:
        for episode_idx in range(args.episodes):
            env = RoboCasaEnv.from_task(_OPEN_DRAWER, 0)
            try:
                logger.info(
                    "robocasa raw episode %s/%s instruction=%r",
                    episode_idx + 1,
                    args.episodes,
                    env.task_description,
                )
                if run_episode(env, vla, args.max_steps):
                    successes += 1
            finally:
                env.close()
    except Exception as exc:
        logger.exception("robocasa raw policy campaign failed")
        print(f"FAIL task={_OPEN_DRAWER!r} error={exc!r}")
        return 1

    total = args.episodes
    rate = successes / total
    print(
        f"PASS task={_OPEN_DRAWER!r} baseline successes={successes}/{total} "
        f"rate={rate:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
