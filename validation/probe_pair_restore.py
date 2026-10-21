"""Probe SimEnv checkpoint restore fidelity for paired M4 arm-2 starts.

Measurement-only: no VLA or LLM. Always exits 0. Writes pair_restore_probe.json.
Colab acceptance requires verdict EXACT (see ``colab_acceptance`` in the JSON).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import config  # noqa: E402
from envs.robocasa_env import expected_action_dim  # noqa: E402
from envs.sim_env import SimEnv  # noqa: E402

_TASK = "open_close_drawer"
_EEF_SLICE = slice(1, 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = os.environ.get("ARTIFACT_DIR", "artifacts/probe_pair_restore")
    parser.add_argument("--artifact-dir", default=default_dir)
    parser.add_argument("--script-steps", type=int, default=50)
    parser.add_argument("--dirty-steps", type=int, default=300)
    return parser.parse_args()


def _script(seed: int, n: int) -> list[list[float]]:
    """Random eef-position actions; other action dims stay zero."""
    dim = expected_action_dim()
    rng = np.random.default_rng(seed)
    out: list[list[float]] = []
    for _ in range(n):
        action = [0.0] * dim
        noise = rng.normal(0.0, 0.3, size=3)
        action[_EEF_SLICE] = np.clip(noise, -1.0, 1.0).tolist()
        out.append(action)
    return out


def _observe_snapshot(env: SimEnv) -> dict[str, object]:
    """Hash live observables at the current env step (no policy calls)."""
    images, state = env.observe()
    return {
        "state": list(state),
        "png_hash": hashlib.sha256(images[0]).hexdigest(),
        "joints": env.drawer_joint_state(),
    }


def _trace(env: SimEnv, actions: list[list[float]]) -> list[dict[str, object]]:
    """Step env and record state, primary PNG hash, and drawer joints."""
    trace: list[dict[str, object]] = []
    for action in actions:
        if env.step(action):
            raise RuntimeError("probe episode terminated early")
        trace.append(_observe_snapshot(env))
    return trace


def _compare(ref: dict[str, object] | list[dict[str, object]],
             other: dict[str, object] | list[dict[str, object]],
             *,
             list_mode: bool) -> dict[str, object]:
    """Compare snapshots step-wise (list) or one pair (dict)."""
    if not list_mode:
        ref_list = [ref]
        other_list = [other]
    else:
        ref_list = ref
        other_list = other
    n = min(len(ref_list), len(other_list))
    first_div: int | None = None
    max_state = 0.0
    png_mismatches = 0
    max_joint = 0.0
    for i in range(n):
        ref_state = ref_list[i]["state"]
        other_state = other_list[i]["state"]
        step_max = max(
            (abs(float(a) - float(b))
             for a, b in zip(ref_state, other_state)),
            default=0.0)
        max_state = max(max_state, step_max)
        if step_max > 1e-9 and first_div is None:
            first_div = i
        if ref_list[i]["png_hash"] != other_list[i]["png_hash"]:
            png_mismatches += 1
        ref_joints = ref_list[i]["joints"]
        other_joints = other_list[i]["joints"]
        for key in set(ref_joints) | set(other_joints):
            jd = abs(float(ref_joints.get(key, 0.0))
                   - float(other_joints.get(key, 0.0)))
            max_joint = max(max_joint, jd)
    return {
        "first_diverging_step": first_div,
        "max_abs_state_diff": max_state,
        "png_hash_mismatches": png_mismatches,
        "max_abs_joint_diff": max_joint,
    }


def _exact(cmp_: dict[str, object]) -> bool:
    return (cmp_["first_diverging_step"] is None
            and cmp_["png_hash_mismatches"] == 0
            and cmp_["max_abs_joint_diff"] == 0.0)


def _verdict(clean: dict[str, object], dirty: dict[str, object],
             post_b: dict[str, object], post_c: dict[str, object]) -> str:
    if not _exact(post_b) or not _exact(post_c):
        return "INEXACT"
    if _exact(clean) and _exact(dirty):
        return "EXACT"
    if _exact(clean) and not _exact(dirty):
        return "CLEAN_ONLY"
    return "INEXACT"


def main() -> int:
    args = _parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    k = args.script_steps
    dirty_n = args.dirty_steps
    config.SAVE_VIDEO = False
    env = SimEnv(_TASK, 0)
    try:
        ckpt = env.capture_episode_checkpoint()
        initial = _observe_snapshot(env)
        script = _script(0, k)
        trace_a = _trace(env, script)
        env.restore_episode_checkpoint(ckpt)
        post_b = _compare(initial, _observe_snapshot(env), list_mode=False)
        trace_b = _trace(env, script)
        _trace(env, _script(1, dirty_n))
        env.restore_episode_checkpoint(ckpt)
        post_c = _compare(initial, _observe_snapshot(env), list_mode=False)
        trace_c = _trace(env, script)
        cmp_b = _compare(trace_a, trace_b, list_mode=True)
        cmp_c = _compare(trace_a, trace_c, list_mode=True)
        verdict = _verdict(cmp_b, cmp_c, post_b, post_c)
        payload = {
            "verdict": verdict,
            "colab_acceptance": verdict == "EXACT",
            "script_steps": k,
            "dirty_steps": dirty_n,
            "artifact_dir": str(artifact_dir),
            "post_restore_B_vs_initial": post_b,
            "post_restore_C_vs_initial": post_c,
            "B_vs_A": cmp_b,
            "C_vs_A": cmp_c,
        }
        out_path = artifact_dir / "pair_restore_probe.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"post_restore_B_vs_initial "
              f"first_diverging_step={post_b['first_diverging_step']} "
              f"max_abs_state_diff={post_b['max_abs_state_diff']} "
              f"png_hash_mismatches={post_b['png_hash_mismatches']} "
              f"max_abs_joint_diff={post_b['max_abs_joint_diff']}")
        print(f"post_restore_C_vs_initial "
              f"first_diverging_step={post_c['first_diverging_step']} "
              f"max_abs_state_diff={post_c['max_abs_state_diff']} "
              f"png_hash_mismatches={post_c['png_hash_mismatches']} "
              f"max_abs_joint_diff={post_c['max_abs_joint_diff']}")
        print(f"B_vs_A first_diverging_step={cmp_b['first_diverging_step']} "
              f"max_abs_state_diff={cmp_b['max_abs_state_diff']} "
              f"png_hash_mismatches={cmp_b['png_hash_mismatches']} "
              f"max_abs_joint_diff={cmp_b['max_abs_joint_diff']}")
        print(f"C_vs_A first_diverging_step={cmp_c['first_diverging_step']} "
              f"max_abs_state_diff={cmp_c['max_abs_state_diff']} "
              f"png_hash_mismatches={cmp_c['png_hash_mismatches']} "
              f"max_abs_joint_diff={cmp_c['max_abs_joint_diff']}")
        print(f"verdict={verdict}")
        print(f"colab_acceptance={payload['colab_acceptance']}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
