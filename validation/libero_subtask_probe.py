"""Live validation probe for ``LiberoEnv.check_subtask`` against real LIBERO sim.

Builds positive and negative world states for ``place``, ``close``, and
``turn_on`` steps, then checks that ``check_subtask(step)`` matches the
constructed predicate truth. Linux + LIBERO only; no VLA service required.

Usage:
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    python validation/libero_subtask_probe.py
    python validation/libero_subtask_probe.py --suite libero_goal --task 0 --repeat 3
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

import config
from libero_env import LiberoEnv
from utils import configure_logging

SKILL_ORDER = ("place", "close", "turn_on")
PLACE_POSITIONS = {
    "in": "inside",
    "on": "on",
}


@dataclass(frozen=True)
class SubtaskCase:
    """One structured subtask check derived from a BDDL goal conjunct."""

    name: str
    skill: str
    step: dict[str, str]
    predicate: str
    predicate_args: tuple[str, ...]


@dataclass
class TrialResult:
    """Outcome for one polarity/repeat attempt."""

    case_name: str
    skill: str
    polarity: str
    repeat: int
    expected: bool
    setup_ok: bool
    setup_predicate: bool | None
    check_subtask: bool | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        if not self.setup_ok:
            return False
        return self.check_subtask is self.expected


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the subtask oracle probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--init-state-idx", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=config.SETTLE_STEPS,
        help="no-op steps after each state reset/setup",
    )
    parser.add_argument(
        "--require-skills",
        default=",".join(SKILL_ORDER),
        help="comma-separated skills that must appear in discovered cases",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exit non-zero on any mismatch or setup failure (default: true)",
    )
    return parser.parse_args()


def _sim_gt(env: LiberoEnv):
    """Return the inner LIBERO problem env."""
    return env._get_sim_gt()


def _forward(env: LiberoEnv) -> None:
    """Refresh derived sim state after direct qpos edits."""
    _sim_gt(env).sim.forward()


def _settle(env: LiberoEnv, settle_steps: int) -> None:
    """Run idle actions so contacts settle after teleport/joint edits."""
    for _ in range(settle_steps):
        env.step(config.SETTLE_ACTION)


def _teleport_object(
    env: LiberoEnv,
    object_name: str,
    pos_xyz: np.ndarray,
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    """Teleport a free-joint object by writing its joint qpos."""
    gt = _sim_gt(env)
    obj = gt.get_object(object_name)
    if not obj.joints:
        raise ValueError(f"object {object_name!r} has no joints to teleport")
    gt.sim.data.set_joint_qpos(
        obj.joints[0],
        np.array([*pos_xyz, *quat_wxyz], dtype=np.float64),
    )
    _forward(env)


def _set_articulation_qpos(env: LiberoEnv, target_name: str, qpos: float) -> None:
    """Set all joints on a target object/site to one scalar qpos."""
    state = _sim_gt(env).object_states_dict[target_name]
    state.set_joint(qpos)
    _forward(env)


def _predicate_truth(env: LiberoEnv, case: SubtaskCase) -> bool:
    """Evaluate the underlying LIBERO predicate for one case."""
    return bool(env.eval_predicate(case.predicate, *case.predicate_args))


def _force_unary_articulation(
    env: LiberoEnv,
    case: SubtaskCase,
    *,
    desired: bool,
) -> None:
    """Search joint values until the unary predicate matches ``desired``."""
    target = case.predicate_args[0]
    samples = np.linspace(-3.14, 3.14, num=40)
    for qpos in samples:
        _set_articulation_qpos(env, target, float(qpos))
        if _predicate_truth(env, case) is desired:
            return
    raise RuntimeError(
        f"could not force {case.predicate}({target!r}) to {desired}; "
        f"tried {len(samples)} joint samples"
    )


def _force_place_predicate(
    env: LiberoEnv,
    case: SubtaskCase,
    *,
    desired: bool,
) -> None:
    """Force an ``in``/``on`` place predicate to the requested truth value."""
    obj_name, target_name = case.predicate_args
    if desired:
        if _predicate_truth(env, case):
            return
        target_pos = (
            _sim_gt(env).object_states_dict[target_name].get_geom_state()["pos"]
        )
        offset = np.array([0.0, 0.0, 0.05], dtype=np.float64)
        _teleport_object(env, obj_name, target_pos + offset)
        if _predicate_truth(env, case):
            return
        raise RuntimeError(
            f"could not force {case.predicate}({obj_name!r}, {target_name!r}) "
            "to True after teleport-to-target"
        )
    _teleport_object(env, obj_name, np.array([10.0, 10.0, 10.0], dtype=np.float64))
    if _predicate_truth(env, case):
        raise RuntimeError(
            f"expected {case.predicate}({obj_name!r}, {target_name!r}) to be False "
            "after far teleport"
        )


def _apply_desired_state(
    env: LiberoEnv,
    case: SubtaskCase,
    *,
    desired: bool,
    init_state,
    settle_steps: int,
) -> None:
    """Reset to canonical init, then mutate sim state toward ``desired``."""
    env._obs = env._env.set_init_state(init_state)
    _settle(env, settle_steps)
    if case.skill == "place":
        _force_place_predicate(env, case, desired=desired)
        return
    _force_unary_articulation(env, case, desired=desired)


def build_cases(env: LiberoEnv) -> list[SubtaskCase]:
    """Build subtask cases from the task BDDL goal conjuncts."""
    cases: list[SubtaskCase] = []
    for index, conjunct in enumerate(env.goal_conjuncts()):
        predicate = conjunct[0].lower()
        if predicate in PLACE_POSITIONS:
            cases.append(
                SubtaskCase(
                    name=f"place_{predicate}_goal_{index}",
                    skill="place",
                    step={
                        "skill": "place",
                        "object_name": conjunct[1],
                        "target_name": conjunct[2],
                        "position": PLACE_POSITIONS[predicate],
                    },
                    predicate=predicate,
                    predicate_args=(conjunct[1], conjunct[2]),
                )
            )
            continue
        if predicate == "close":
            cases.append(
                SubtaskCase(
                    name=f"close_goal_{index}",
                    skill="close",
                    step={"skill": "close", "target_name": conjunct[1]},
                    predicate="close",
                    predicate_args=(conjunct[1],),
                )
            )
            continue
        if predicate == "turnon":
            cases.append(
                SubtaskCase(
                    name=f"turn_on_goal_{index}",
                    skill="turn_on",
                    step={"skill": "turn_on", "target_name": conjunct[1]},
                    predicate="turnon",
                    predicate_args=(conjunct[1],),
                )
            )
    return cases


def _required_skills(raw: str) -> tuple[str, ...]:
    """Parse and validate the required skill list."""
    skills = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [skill for skill in skills if skill not in SKILL_ORDER]
    if invalid:
        raise ValueError(
            f"unknown require-skills entries {invalid}; expected subset of {SKILL_ORDER}"
        )
    return skills


def _missing_required_skills(
    cases: list[SubtaskCase],
    required: tuple[str, ...],
) -> list[str]:
    """Return required skills that have no discovered case."""
    present = {case.skill for case in cases}
    return [skill for skill in required if skill not in present]


def _run_trial(
    env: LiberoEnv,
    case: SubtaskCase,
    *,
    polarity: str,
    repeat: int,
    desired: bool,
    init_state,
    settle_steps: int,
    apply_state: Callable[..., None],
) -> TrialResult:
    """Run one positive/negative trial for a case."""
    try:
        apply_state(
            env,
            case,
            desired=desired,
            init_state=init_state,
            settle_steps=settle_steps,
        )
        setup_predicate = _predicate_truth(env, case)
        setup_ok = setup_predicate is desired
        check_value = env.check_subtask(case.step)
        return TrialResult(
            case_name=case.name,
            skill=case.skill,
            polarity=polarity,
            repeat=repeat,
            expected=desired,
            setup_ok=setup_ok,
            setup_predicate=setup_predicate,
            check_subtask=check_value,
        )
    except Exception as exc:
        return TrialResult(
            case_name=case.name,
            skill=case.skill,
            polarity=polarity,
            repeat=repeat,
            expected=desired,
            setup_ok=False,
            setup_predicate=None,
            check_subtask=None,
            error=str(exc),
        )


def _print_trial(result: TrialResult) -> None:
    """Print one machine-readable trial line."""
    status = "PASS" if result.passed else "FAIL"
    setup_predicate = (
        "null" if result.setup_predicate is None else str(result.setup_predicate).lower()
    )
    check_subtask = (
        "null" if result.check_subtask is None else str(result.check_subtask).lower()
    )
    error = result.error or "-"
    print(
        "trial "
        f"case={result.case_name} "
        f"skill={result.skill} "
        f"polarity={result.polarity} "
        f"repeat={result.repeat} "
        f"expected={str(result.expected).lower()} "
        f"setup_ok={str(result.setup_ok).lower()} "
        f"setup_predicate={setup_predicate} "
        f"check_subtask={check_subtask} "
        f"error={error!r} "
        f"status={status}"
    )


def main() -> int:
    """Run the live subtask oracle probe and print a compact summary."""
    args = parse_args()
    if args.repeat <= 0:
        print("repeat must be positive", file=sys.stderr)
        return 2
    if args.settle_steps < 0:
        print("settle-steps must be >= 0", file=sys.stderr)
        return 2

    try:
        required_skills = _required_skills(args.require_skills)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    configure_logging()

    try:
        probe_env = LiberoEnv(
            args.suite, args.task, init_state_idx=args.init_state_idx
        )
    except ImportError as exc:
        print(f"FAIL: LIBERO unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        task_description = probe_env.task_description
        init_state = probe_env._init_states[probe_env.init_state_idx]
        cases = build_cases(probe_env)
        missing = _missing_required_skills(cases, required_skills)
    finally:
        probe_env.close()

    print(f"suite             {args.suite!r}")
    print(f"task              {args.task}")
    print(f"init_state_idx    {args.init_state_idx}")
    print(f"task_description  {task_description!r}")
    print(f"repeat            {args.repeat}")
    print(f"settle_steps      {args.settle_steps}")
    print(f"require_skills    {','.join(required_skills)}")
    print(f"discovered_cases  {len(cases)}")
    print()

    if missing:
        print(
            "FAIL: task goal does not cover required skills: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Pick another --suite/--task with close/turn_on/in/on goal conjuncts.",
            file=sys.stderr,
        )
        return 2

    results: list[TrialResult] = []
    for repeat in range(args.repeat):
        env = LiberoEnv(args.suite, args.task, init_state_idx=args.init_state_idx)
        try:
            for case in cases:
                for polarity, desired in (("negative", False), ("positive", True)):
                    result = _run_trial(
                        env,
                        case,
                        polarity=polarity,
                        repeat=repeat,
                        desired=desired,
                        init_state=init_state,
                        settle_steps=args.settle_steps,
                        apply_state=_apply_desired_state,
                    )
                    results.append(result)
                    _print_trial(result)
        finally:
            env.close()

    passes = sum(1 for result in results if result.passed)
    failures = len(results) - passes
    setup_failures = sum(1 for result in results if not result.setup_ok)
    mismatches = sum(
        1
        for result in results
        if result.setup_ok and result.check_subtask is not result.expected
    )

    print()
    print(
        "summary "
        f"cases={len(cases)} "
        f"trials={len(results)} "
        f"passes={passes} "
        f"failures={failures} "
        f"setup_failures={setup_failures} "
        f"mismatches={mismatches}"
    )

    if failures == 0:
        return 0
    if args.strict:
        return 1
    if mismatches == 0:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
