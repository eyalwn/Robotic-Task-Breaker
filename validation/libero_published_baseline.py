"""Published OpenPI pi0.5 LIBERO baseline rates and condition-matching checklist.

Official pi0.5 @ 30k finetuned on LIBERO, from:
https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md
Checkpoint: gs://openpi-assets/checkpoints/pi05_libero
"""
from __future__ import annotations

import argparse
import sys

import config

SOURCE_URL = (
    "https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md"
)
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"

PUBLISHED_RATES_PCT = {
    "libero_spatial": 98.8,
    "libero_object": 98.2,
    "libero_goal": 98.0,
    "libero_10": 92.4,
    "average": 96.85,
}


def published_rate(suite: str) -> float:
    """Return the published success rate for ``suite`` as a fraction in [0, 1]."""
    if suite not in PUBLISHED_RATES_PCT:
        known = ", ".join(sorted(PUBLISHED_RATES_PCT))
        raise ValueError(f"unknown suite {suite!r}; known: {known}")
    return PUBLISHED_RATES_PCT[suite] / 100.0


def print_reference() -> None:
    """Print published rates and the condition-matching checklist."""
    print("OpenPI pi0.5 published LIBERO baseline (pi0.5 @ 30k, pi05_libero)")
    print(f"source:     {SOURCE_URL}")
    print(f"checkpoint: {CHECKPOINT}")
    print()
    print("published success rates:")
    for suite, pct in PUBLISHED_RATES_PCT.items():
        print(f"  {suite:16s} {pct:5.2f}%")
    print()
    print("condition-matching checklist (M4/M5 claims):")
    print(f"  resolution:          {config.RESOLUTION}")
    print(f"  seed:                {config.SEED}")
    print(f"  settle steps:        {config.SETTLE_STEPS}")
    print("  wait steps:          align with OpenPI LIBERO eval loop")
    print("  suite max steps:     from config.MAX_EPISODE_STEPS per suite")
    for suite, steps in config.MAX_EPISODE_STEPS.items():
        if suite.startswith("libero_"):
            print(f"    {suite:16s} {steps}")
    print(f"  vla actions/chunk:   {config.VLA_ACTIONS_PER_CHUNK} (service-side)")
    print("  replan horizon:      matches VLA_ACTIONS_PER_CHUNK")
    print("  init-state indexing: episode i -> init_state[i % count]")
    print("  full suite claims:   50 trials/task")
    print('  oracle disclosure:   required when using subtask-completion oracle')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=tuple(PUBLISHED_RATES_PCT.keys()),
        help="suite to compare against published rate",
    )
    parser.add_argument(
        "--ours-rate",
        type=float,
        help="our reported success rate as a fraction (e.g. 0.95 for 95%%)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_reference()

    if args.suite is None:
        return 0

    published = published_rate(args.suite)
    print()
    print(f"comparison suite: {args.suite}")
    print(f"published:        {published * 100:.2f}%")

    if args.ours_rate is None:
        return 0

    ours = args.ours_rate
    print(f"ours:             {ours * 100:.2f}%")
    if ours >= published:
        print("PASS: ours >= published")
        return 0
    print("FAIL: ours < published")
    return 1


if __name__ == "__main__":
    sys.exit(main())
