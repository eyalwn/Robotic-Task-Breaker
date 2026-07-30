"""Phase 1 gate: prove the VLA -> LIBERO loop works, with nothing else in the way.

No planner, no executor, no skills, no plan AST. This is the reference OpenPI
LIBERO evaluation loop against our own env wrapper and policy client, so a
failure here is a failure of the environment, the service, or the checkpoint --
never of the orchestration layers.

Requires the policy service running (``python services/smolvla_service.py``) and,
on a headless machine, ``export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl``.

Usage:
    python validation/libero_probe.py                       # libero_object, task 0
    python validation/libero_probe.py --suite libero_10 --task 0
"""
from __future__ import annotations

import argparse
import math
import sys

import config
from libero_env import LiberoEnv
from utils import configure_logging
from vla_client import VLAClient, VLAError


# XXX adding arguments?? serously?? very misleading. better to hard-code:
# env = LiberoEnv("libero_object", 0)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 uses the suite's longest training demonstration length",
    )
    return parser.parse_args()


def _distance(a: list[float], b: list[float]) -> float:
    return math.dist(a[:3], b[:3])


def probe(env: LiberoEnv, vla: VLAClient, task: str, max_steps: int) -> dict:
    """Run the closed-loop policy until success or the step budget runs out."""
    start_eef = env.state()[:3]
    targets = env.object_names()
    closest = {name: math.inf for name in targets}
    gripper = [env.state()[6]]
    action_sums = [0.0] * 7
    actions_applied = 0
    success = False

    while env.steps < max_steps and not success:
        image, wrist, state = env.observe()
        chunk = vla.predict(image, wrist, state, task)
        for action in chunk[: config.REPLAN_STEPS]:
            success = env.step(action)
            actions_applied += 1
            for i, value in enumerate(action):
                action_sums[i] += abs(value)
            state = env.state()
            gripper.append(state[6])
            for name in targets:
                position = env.body_position(name)
                if position is not None:
                    closest[name] = min(closest[name], _distance(state, position))
            if success or env.steps >= max_steps:
                break

    state = env.state()
    return {
        "success": success,
        "steps": env.steps,
        "actions": actions_applied,
        "eef_displacement": _distance(start_eef, state),
        "closest_approach": closest,
        "gripper_range": (min(gripper), max(gripper)),
        "mean_abs_action": [s / max(actions_applied, 1) for s in action_sums],
    }


def print_summary(task: str, metrics: dict) -> None:
    print()
    print(f"task              {task!r}")
    print(f"steps             {metrics['steps']} ({metrics['actions']} policy actions)")
    print(f"eef displacement  {metrics['eef_displacement']:.4f} m")
    low, high = metrics["gripper_range"]
    print(f"gripper qpos      {low:.4f} .. {high:.4f}")
    means = ", ".join(f"{v:.3f}" for v in metrics["mean_abs_action"])
    print(f"mean |action|     [{means}]")
    for name, distance in sorted(metrics["closest_approach"].items()):
        shown = "n/a" if math.isinf(distance) else f"{distance:.4f} m"
        print(f"closest approach  {name}: {shown}")
    print()
    print("PASS: task solved" if metrics["success"] else "FAIL: task not solved")


def main() -> int:
    args = parse_args()
    configure_logging()

    vla = VLAClient()
    try:
        print(f"vla service       {vla.health()}")
    except VLAError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print("Start it first: python services/smolvla_service.py", file=sys.stderr)
        return 2

    env = LiberoEnv(args.suite, args.task)
    task = env.task_description
    try:
        metrics = probe(env, vla, task, args.max_steps or env.max_steps)
    finally:
        env.close()

    print_summary(task, metrics)
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
