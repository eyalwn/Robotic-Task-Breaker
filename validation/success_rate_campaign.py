"""Monolithic VLA baseline for LIBERO success-rate campaigns.

No planner, executor, skills, or plan AST. Direct policy loop only:
``LiberoEnv`` + ``VLAClient``. Requires the policy service
(``python services/run_vla_server.py``) and on headless Linux:
``export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl``.

Usage:
    python validation/success_rate_campaign.py
    python validation/success_rate_campaign.py --suite libero_10 --task 0 --episodes 10
"""
from __future__ import annotations

import argparse
import sys
import time

import config
from libero_env import LiberoEnv
from utils import configure_logging
from vla_client import VLAClient, VLAError


def parse_args() -> argparse.Namespace:
    """Parse campaign CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--vla-actions-per-chunk",
        type=int,
        default=config.VLA_ACTIONS_PER_CHUNK,
        help=(
            "expected service chunk size (from config.VLA_ACTIONS_PER_CHUNK); "
            "changing it requires restarting the VLA service"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 uses the suite episode cap from config",
    )
    return parser.parse_args()


def run_episode(
    env: LiberoEnv,
    vla: VLAClient,
    task: str,
    max_steps: int,
) -> dict[str, object]:
    """Run one closed-loop episode until success or step budget."""
    success = False
    start = time.perf_counter()
    while env.steps < max_steps and not success:
        image, wrist, state = env.observe()
        chunk = vla.predict(image, wrist, state, task)
        for action in chunk:
            success = env.step(action)
            if success or env.steps >= max_steps:
                break
    return {
        "success": success,
        "steps": env.steps,
        "elapsed_s": time.perf_counter() - start,
    }


def main() -> int:
    """Run a multi-episode campaign and print per-episode plus aggregate results."""
    args = parse_args()
    if args.episodes <= 0:
        print("episodes must be positive", file=sys.stderr)
        return 2
    configure_logging()

    vla = VLAClient()
    try:
        print(f"vla service       {vla.health()}")
    except VLAError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print("Start it first: python services/run_vla_server.py", file=sys.stderr)
        return 2

    probe_env = LiberoEnv(args.suite, args.task, init_state_idx=0)
    task = probe_env.task_description
    init_count = probe_env.init_state_count()
    max_steps = args.max_steps or probe_env.max_steps
    probe_env.close()

    print(f"suite             {args.suite!r}")
    print(f"task              {args.task}")
    print(f"task_description  {task!r}")
    print(f"episodes          {args.episodes}")
    print(f"init_states       {init_count}")
    print(f"max_steps         {max_steps}")
    print(f"vla_actions_chunk {config.VLA_ACTIONS_PER_CHUNK} (service-owned)")
    if args.vla_actions_per_chunk != config.VLA_ACTIONS_PER_CHUNK:
        print(
            f"note: --vla-actions-per-chunk={args.vla_actions_per_chunk} ignored; "
            f"service uses config.VLA_ACTIONS_PER_CHUNK={config.VLA_ACTIONS_PER_CHUNK}",
            file=sys.stderr,
        )
    print()

    successes = 0
    for episode in range(args.episodes):
        init_idx = LiberoEnv.init_state_idx_for_episode(episode, init_count)
        env = LiberoEnv(args.suite, args.task, init_state_idx=init_idx)
        try:
            result = run_episode(env, vla, task, max_steps)
        finally:
            env.close()
        if result["success"]:
            successes += 1
        print(
            f"episode {episode:3d}  init={init_idx:2d}  "
            f"success={str(result['success']).lower():5s}  "
            f"steps={result['steps']:4d}  "
            f"elapsed={result['elapsed_s']:.1f}s"
        )

    rate = successes / args.episodes
    print()
    print(f"success_rate      {successes}/{args.episodes} ({rate:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
