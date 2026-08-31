"""Prompt-sweep probe for single-arm control on one LIBERO task.

This isolates whether failures come from the arm-task text, the per-skill
step cap, or both. It runs the same closed-loop control pattern as executor
arm skills: query policy, execute the full returned chunk, stop on task
success or step budget.

Usage:
    python validation/arm_prompt_probe.py
    python validation/arm_prompt_probe.py --skill-step-budgets 120 280
"""
from __future__ import annotations

import argparse
import sys
import time

import config
from libero_env import LiberoEnv
from utils import configure_logging
from vla_client import VLAClient, VLAError

DEFAULT_PROMPTS = (
    "{task_description}",
    "move the alphabet soup to the basket",
    "pick up the alphabet soup",
    "put the alphabet soup in the basket",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the prompt-sweep probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--init-state-idx", type=int, default=0)
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
            "Use {task_description} placeholder if needed."
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


def _resolve_prompts(task_description: str, extra_prompts: list[str]) -> list[str]:
    """Build ordered prompt variants for the sweep."""
    prompt_templates = list(DEFAULT_PROMPTS)
    prompt_templates.extend(extra_prompts)
    prompts: list[str] = []
    seen: set[str] = set()
    for template in prompt_templates:
        rendered = template.format(task_description=task_description).strip()
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
    max_skill_steps: int,
) -> dict[str, object]:
    """Run one prompt with executor-like control and return metrics."""
    success = False
    steps_at_start = env.steps
    policy_calls = 0
    started = time.perf_counter()

    while (
        env.steps - steps_at_start < max_skill_steps
        and env.steps < env.max_steps
        and not success
    ):
        image, wrist, state = env.observe()
        chunk = vla.predict(image, wrist, state, task)
        policy_calls += 1
        for action in chunk:
            if env.steps - steps_at_start >= max_skill_steps or env.steps >= env.max_steps:
                break
            success = env.step(action)
            if success:
                break

    run_steps = env.steps - steps_at_start
    return {
        "success": success,
        "run_steps": run_steps,
        "total_steps": env.steps,
        "policy_calls": policy_calls,
        "elapsed_s": time.perf_counter() - started,
        "hit_skill_budget": run_steps >= max_skill_steps,
        "hit_episode_budget": env.steps >= env.max_steps,
    }


def main() -> int:
    """Run the prompt sweep and print one-line outcomes."""
    args = parse_args()
    configure_logging()

    vla = VLAClient()
    try:
        print(f"vla service       {vla.health()}")
    except VLAError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print("Start it first: python services/smolvla_service.py", file=sys.stderr)
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
    prompts = _resolve_prompts(task_description, args.prompt)

    print(f"suite             {args.suite!r}")
    print(f"task              {args.task}")
    print(f"init_state_idx    {args.init_state_idx}")
    print(f"task_description  {task_description!r}")
    print(f"episode_cap       {episode_cap}")
    print(f"vla_actions_chunk {config.VLA_ACTIONS_PER_CHUNK} (service-owned)")
    if args.vla_actions_per_chunk != config.VLA_ACTIONS_PER_CHUNK:
        print(
            f"note: --vla-actions-per-chunk={args.vla_actions_per_chunk} ignored; "
            f"service uses config.VLA_ACTIONS_PER_CHUNK={config.VLA_ACTIONS_PER_CHUNK}",
            file=sys.stderr,
        )
    print(f"step_budgets      {budgets}")
    print()

    for budget in budgets:
        print("=" * 96)
        print(f"budget={budget}")
        print("=" * 96)
        for idx, prompt in enumerate(prompts, start=1):
            env = LiberoEnv(args.suite, args.task, init_state_idx=args.init_state_idx)
            try:
                result = run_one_prompt(
                    env,
                    vla,
                    task=prompt,
                    max_skill_steps=budget,
                )
            finally:
                env.close()
            print(
                f"[{idx:02d}/{len(prompts):02d}] "
                f"success={str(result['success']).lower():5s} "
                f"run_steps={result['run_steps']:3d} "
                f"total_steps={result['total_steps']:3d} "
                f"policy_calls={result['policy_calls']:3d} "
                f"hit_skill_budget={str(result['hit_skill_budget']).lower():5s} "
                f"hit_episode_budget={str(result['hit_episode_budget']).lower():5s} "
                f"elapsed={result['elapsed_s']:.1f}s "
                f"prompt={prompt!r}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
