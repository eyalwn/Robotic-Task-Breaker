"""M3 feasibility probe: can we reach a RoboCasa composite task through the
GR00T PandaOmron gym wrapper, and does it expose a usable subtask oracle?

This is a one-off diagnostic for M3 suite selection, not a gate and not wired
into runtime. The RoboCasa sim Python lives under the **Isaac-GR00T** checkout
(``/content/Isaac-GR00T`` on Colab), not under the Longor repo.

After ``colab/run_longor_task.ipynb`` section 3, use the paths written to
``.colab_m2_env``::

    set -a && source /content/Longor/.colab_m2_env && set +a
    "${ROBO_PY}" /content/Longor/validation/probe_robocasa_composite_support.py \\
        --task KettleBoiling

Equivalent absolute path (if section 3 completed with default ``GROOT_ROOT``)::

    /content/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        /content/Longor/validation/probe_robocasa_composite_support.py --task KettleBoiling

It tries the GR00T PandaOmron-wrapped env id first, then falls back to the
raw upstream RoboCasa gym id, and reports observation keys plus any
method/attribute names that look like a per-stage success signal.
"""
from __future__ import annotations

import argparse
import json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="KettleBoiling",
        help="RoboCasa task class name, e.g. KettleBoiling.",
    )
    return parser.parse_args()


def _make_env(gym_module, env_id: str):
    """Return (env, error) for one gym.make attempt."""
    try:
        return gym_module.make(env_id, enable_render=True), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _subtask_oracle_candidates(env) -> list[str]:
    """List unwrapped-sim method/attribute names that look like a per-stage
    success or subtask-completion check."""
    sim_env = getattr(getattr(env, "unwrapped", None), "env", None)
    if sim_env is None:
        return []
    keywords = ("success", "subtask", "stage", "phase", "progress")
    return sorted(
        name for name in dir(sim_env)
        if any(keyword in name.lower() for keyword in keywords)
    )


def _probe_env_id(gym_module, env_id: str) -> dict:
    env, error = _make_env(gym_module, env_id)
    if env is None:
        return {"env_id": env_id, "registered": False, "error": error}
    try:
        obs, _info = env.reset()
        obs_keys = sorted(str(key) for key in obs) if isinstance(obs, dict) else None
        return {
            "env_id": env_id,
            "registered": True,
            "obs_keys": obs_keys,
            "subtask_oracle_candidates": _subtask_oracle_candidates(env),
        }
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    import gymnasium as gym
    import robocasa  # noqa: F401 - registers gym envs on import
    import robosuite  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

    candidates = (
        f"robocasa_panda_omron/{args.task}_PandaOmron_Env",
        f"robocasa/{args.task}",
    )
    reports = [_probe_env_id(gym, env_id) for env_id in candidates]
    print(json.dumps(reports, indent=2))
    return 0 if any(report["registered"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
