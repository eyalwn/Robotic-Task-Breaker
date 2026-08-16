"""Prompt-sweep probe for one ``grab`` skill on one LIBERO task.

This isolates whether an executor ``grab`` step fails because of the VLA task
text, the per-skill step cap, or both. Each run uses the same closed-loop
control pattern as the executor (query policy, execute the full returned chunk)
and is scored on ``LiberoEnv.check_subtask`` for the grab, not on full-task
success -- a "pick up X" prompt can never satisfy a multi-step task goal.

The first prompt (``{task_description}``) is the control: if it cannot grasp,
the whole sweep is uninformative and the probe fails.

Usage:
    python validation/arm_prompt_probe.py
    python validation/arm_prompt_probe.py --skill-step-budgets 120 280
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import config
from libero_env import LiberoEnv
from utils import configure_logging
from vla_client import VLAClient, VLAError

GRIPPER_INDEX = 6
DEFAULT_PROMPTS = (
    "{task_description}",
    "pick up the {object}",
    "pick up the {object} and place it in the basket",
    "put the {object} in the basket",
    "grasp the {object}",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the prompt-sweep probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--init-state-idx", type=int, default=0)
    parser.add_argument(
        "--object",
        default="alphabet soup",
        help="object the grab step must end up holding (scoring target)",
    )
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
        "--skill-step-budgets",
        type=int,
        nargs="*",
        default=None,
        help=(
            "per-run step caps measured from first policy action "
            "(default: MAX_SKILL_STEPS and suite episode cap)"
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help=(
            "additional prompt variant; can be provided multiple times. "
            "Use {task_description} and {object} placeholders if needed."
        ),
    )
    return parser.parse_args()


def _resolve_budgets(
    raw_budgets: list[int] | None,
    *,
    default_skill_steps: int,
    episode_cap: int,
) -> list[int]:
    """Resolve and validate budget list while keeping order and uniqueness."""
    budgets = raw_budgets or [default_skill_steps, episode_cap]
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for value in budgets:
        if value <= 0:
            raise ValueError(f"skill-step-budgets must be positive, got {value}")
        if value in seen:
            continue
        seen.add(value)
        ordered_unique.append(value)
    return ordered_unique


def _resolve_prompts(
    task_description: str,
    object_name: str,
    extra_prompts: list[str],
) -> list[str]:
    """Build ordered prompt variants for the sweep, control first."""
    templates = [*DEFAULT_PROMPTS, *extra_prompts]
    prompts: list[str] = []
    seen: set[str] = set()
    for template in templates:
        rendered = template.format(
            task_description=task_description, object=object_name
        ).strip()
        if not rendered or rendered in seen:
            continue
        seen.add(rendered)
        prompts.append(rendered)
    return prompts


def run_one_prompt(
    env: LiberoEnv,
    vla: VLAClient,
    *,
    task: str,
    step: dict[str, str],
    max_skill_steps: int,
) -> dict[str, object]:
    """Run one prompt with executor-like control until the grab succeeds."""
    steps_at_start = env.steps
    grasped = task_success = False
    policy_calls = 0
    min_distance, step_at_min, max_grip_cmd = math.inf, 0, -math.inf
    started = time.perf_counter()

    def out_of_budget() -> bool:
        return (
            env.steps - steps_at_start >= max_skill_steps
            or env.steps >= env.max_steps
        )

    while not grasped and not out_of_budget():
        image, wrist, state = env.observe()
        chunk = vla.predict(image, wrist, state, task)
        policy_calls += 1
        for action in chunk:
            if out_of_budget():
                break
            max_grip_cmd = max(max_grip_cmd, action[GRIPPER_INDEX])
            task_success = env.step(action) or task_success
            distance = env.eef_goal_distance(step)
            if distance < min_distance:
                min_distance = distance
                step_at_min = env.steps - steps_at_start
            grasped = env.check_subtask(step)
            if grasped:
                break

    run_steps = env.steps - steps_at_start
    return {
        "grasped": grasped,
        "task_success": task_success,
        "min_distance": min_distance,
        "step_at_min": step_at_min,
        "max_grip_cmd": max_grip_cmd,
        "run_steps": run_steps,
        "policy_calls": policy_calls,
        "elapsed_s": time.perf_counter() - started,
        "hit_skill_budget": run_steps >= max_skill_steps,
        "hit_episode_budget": env.steps >= env.max_steps,
    }


def _print_run(index: int, total: int, prompt: str, result: dict[str, object]) -> None:
    """Print one machine-readable sweep line."""
    print(
        f"[{index:02d}/{total:02d}] "
        f"grasped={str(result['grasped']).lower():5s} "
        f"task_ok={str(result['task_success']).lower():5s} "
        f"min_eef={result['min_distance']:.4f} "
        f"at_step={result['step_at_min']:3d} "
        f"max_grip={result['max_grip_cmd']:+.3f} "
        f"run_steps={result['run_steps']:3d} "
        f"calls={result['policy_calls']:3d} "
        f"hit_skill_budget={str(result['hit_skill_budget']).lower():5s} "
        f"elapsed={result['elapsed_s']:.1f}s "
        f"prompt={prompt!r}"
    )


def main() -> int:
    """Run the prompt sweep and print a per-prompt table plus a verdict."""
    args = parse_args()
    configure_logging()

    vla = VLAClient()
    try:
        print(f"vla service       {vla.health()}")
    except VLAError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print("Start it first: python services/run_vla_server.py", file=sys.stderr)
        return 2

    probe_env = LiberoEnv(args.suite, args.task, init_state_idx=args.init_state_idx)
    try:
        task_description = probe_env.task_description
        episode_cap = probe_env.max_steps
    finally:
        probe_env.close()

    try:
        budgets = _resolve_budgets(
            args.skill_step_budgets,
            default_skill_steps=config.MAX_SKILL_STEPS,
            episode_cap=episode_cap,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    prompts = _resolve_prompts(task_description, args.object, args.prompt)
    step = {"skill": "grab", "object_name": args.object}

    print(f"suite             {args.suite!r}")
    print(f"task              {args.task}")
    print(f"init_state_idx    {args.init_state_idx}")
    print(f"task_description  {task_description!r}")
    print(f"grab_object       {args.object!r}")
    print(f"episode_cap       {episode_cap}")
    print(f"vla_actions_chunk {config.VLA_ACTIONS_PER_CHUNK} (service-owned)")
    if args.vla_actions_per_chunk != config.VLA_ACTIONS_PER_CHUNK:
        print(
            f"note: --vla-actions-per-chunk={args.vla_actions_per_chunk} ignored; "
            f"service uses config.VLA_ACTIONS_PER_CHUNK={config.VLA_ACTIONS_PER_CHUNK}",
            file=sys.stderr,
        )
    print(f"step_budgets      {budgets}")
    print(f"control_prompt    {prompts[0]!r}")
    print()

    control_grasped = False
    for budget in budgets:
        print("=" * 96)
        print(f"budget={budget}")
        print("=" * 96)
        grasped_count = 0
        for index, prompt in enumerate(prompts, start=1):
            env = LiberoEnv(args.suite, args.task, init_state_idx=args.init_state_idx)
            try:
                result = run_one_prompt(
                    env,
                    vla,
                    task=prompt,
                    step=step,
                    max_skill_steps=budget,
                )
            finally:
                env.close()
            grasped_count += bool(result["grasped"])
            control_grasped = control_grasped or (
                index == 1 and bool(result["grasped"])
            )
            _print_run(index, len(prompts), prompt, result)
        print(f"grasped_prompts   {grasped_count}/{len(prompts)}")
        print()

    verdict = "PASS" if control_grasped else "FAIL"
    print(f"control_grasped   {str(control_grasped).lower()}")
    print(f"verdict           {verdict}")
    if not control_grasped:
        print(
            "FAIL: the task_description control never grasped, so prompt "
            "comparisons are meaningless. Try another --init-state-idx or a "
            "larger budget.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
