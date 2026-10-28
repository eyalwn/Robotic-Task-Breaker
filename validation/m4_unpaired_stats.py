"""M4 unpaired Robotic Task Breaker vs bare stats (one-sided Fisher exact).

Compares success counts with predeclared interim (20/arm) and final (50/arm)
thresholds from ``SRD.md``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_INTERIM_N = 20
_FINAL_N = 50
_INTERIM_P = 0.01
_FINAL_P = 0.04


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("interim", "final"), required=True,
        help="interim=20/arm thresholds; final=50/arm " "thresholds.")
    parser.add_argument("--robotic-task-breaker-summary", type=Path)
    parser.add_argument("--bare-summary", type=Path)
    parser.add_argument("--robotic-task-breaker-successes", type=int)
    parser.add_argument("--robotic-task-breaker-total", type=int)
    parser.add_argument("--bare-successes", type=int)
    parser.add_argument("--bare-total", type=int)
    parser.add_argument("--output", type=Path,
                        help="Optional path for machine-readable JSON verdict.")
    return parser.parse_args()


def _load_summary(path: Path | None) -> tuple[int, int]:
    if path is None:
        raise ValueError("summary path is required when counts are omitted")
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["successes"]), int(data["episodes"])


def _counts_from_args(args: argparse.Namespace) -> tuple[int, int, int, int]:
    summary = args.robotic_task_breaker_summary
    successes = args.robotic_task_breaker_successes
    total = args.robotic_task_breaker_total
    if summary is not None:
        arm_succ, arm_total = _load_summary(summary)
    elif successes is not None and total is not None:
        arm_succ, arm_total = successes, total
    else:
        raise SystemExit(
            "Provide --robotic-task-breaker-summary or both "
            "--robotic-task-breaker-successes and "
            "--robotic-task-breaker-total")
    if args.bare_summary is not None:
        bare_succ, bare_total = _load_summary(args.bare_summary)
    elif args.bare_successes is not None and args.bare_total is not None:
        bare_succ, bare_total = args.bare_successes, args.bare_total
    else:
        raise SystemExit(
            "Provide --bare-summary or both --bare-successes and --bare-total")
    return arm_succ, arm_total, bare_succ, bare_total


def _hypergeom_pmf(k: int, n: int, K: int, N: int) -> float:
    """P(X=k) for X ~ Hypergeom(N, K, n)."""
    if k < 0 or k > n or K < 0 or K > N or n > N:
        return 0.0
    return (math.comb(K, k) * math.comb(N - K, n - k)) / math.comb(N, n)


def fisher_exact_greater(
        arm_succ: int, arm_total: int, bare_succ: int,
        bare_total: int) -> float:
    """One-sided Fisher exact p-value for Robotic Task Breaker > bare."""
    a = arm_succ
    b = arm_total - arm_succ
    c = bare_succ
    d = bare_total - bare_succ
    row1 = a + b
    row2 = c + d
    col1 = a + c
    n = row1 + row2
    p_obs = _hypergeom_pmf(a, row1, col1, n)
    p_value = 0.0
    low = max(0, col1 - row2)
    high = min(row1, col1)
    for a_i in range(low, high + 1):
        b_i = row1 - a_i
        c_i = col1 - a_i
        d_i = row2 - c_i
        if b_i < 0 or c_i < 0 or d_i < 0:
            continue
        p_i = _hypergeom_pmf(a_i, row1, col1, n)
        if a_i >= a and p_i <= p_obs + 1e-15:
            p_value += p_i
    return min(1.0, p_value)


def evaluate(
        arm_succ: int, arm_total: int, bare_succ: int, bare_total: int,
        mode: str) -> dict[str, object]:
    """Return verdict payload for interim or final M4 look."""
    expected_n = _INTERIM_N if mode == "interim" else _FINAL_N
    p_cutoff = _INTERIM_P if mode == "interim" else _FINAL_P
    sr_arm = arm_succ / arm_total if arm_total else 0.0
    sr_bare = bare_succ / bare_total if bare_total else 0.0
    p_value = fisher_exact_greater(arm_succ, arm_total, bare_succ, bare_total)
    claim = (p_value <= p_cutoff and sr_arm > sr_bare
             and arm_total >= expected_n and bare_total >= expected_n)
    return {
        "mode": mode, "expected_episodes_per_arm": expected_n,
        "p_threshold": p_cutoff,
        "robotic_task_breaker": {"successes": arm_succ, "episodes": arm_total,
                                 "success_rate": sr_arm},
        "bare": {"successes": bare_succ, "episodes": bare_total,
                 "success_rate": sr_bare},
        "p_value_one_sided": p_value, "claim": claim,
        "verdict": "claim" if claim else "no_claim"}


def main() -> int:
    args = _parse_args()
    arm_succ, arm_total, bare_succ, bare_total = _counts_from_args(args)
    result = evaluate(arm_succ, arm_total, bare_succ, bare_total, args.mode)
    rate = result["robotic_task_breaker"]["success_rate"]
    line = (f"M4 {result['mode']}: {result['verdict']} "
            f"(p={result['p_value_one_sided']:.6f}, "
            f"SR_robotic_task_breaker={rate:.3f}, "
            f"SR_bare={result['bare']['success_rate']:.3f})")
    print(line)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
