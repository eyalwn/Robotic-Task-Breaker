"""M3 KettleBoiling primitive gate: fixed two-step plan, no LLM.

Runs place then turn_on inside the KettleBoiling scene with per-stage oracle
checks. Colab: after ``run_longor_task.ipynb`` section 3, source
``.colab_m2_env`` and point ``ROBO_PY`` at this script (VLA HTTP on :8765).

Burner choice uses sim ground truth when ``--burner auto`` (default): the gate
isolates primitive viability; the real Longor loop has the LLM choose from the
snapshot.

Measurement-only script: always exits 0 regardless of episode outcomes.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import config  # noqa: E402
from envs.robocasa_env import burner_phrase_to_key  # noqa: E402
from executor import Executor, StopReason  # noqa: E402
from envs.sim_env import SimEnv  # noqa: E402

_DISTR_OCCUPY_RADIUS = 0.20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--burner",
        default="auto",
        help=(
            "Burner phrase for every episode, or 'auto' to pick a free burner "
            "from sim layout after each reset."
        ),
    )
    return parser.parse_args()


def _burner_key_to_phrase(key: str) -> str:
    return f"{key.replace('_', ' ')} burner of the stove"


def _body_xy(sim_env, obj) -> tuple[float, float]:
    body_id = sim_env.obj_body_id[obj.name]
    pos = sim_env.sim.data.body_xpos[body_id]
    return float(pos[0]), float(pos[1])


def _site_xy(sim_env, site) -> tuple[float, float]:
    """XY of a burner site. RoboCasa stores these as XML elements, not dicts."""
    pos = sim_env.sim.data.get_site_xpos(site.get("name"))
    return float(pos[0]), float(pos[1])


def _choose_burner_phrase(backend) -> str | None:
    """Return a free burner phrase from sim layout, or None if none are free."""
    sim_env = backend._sim_env()
    sites = sim_env.stove.burner_sites
    valid_keys = tuple(sorted(sites.keys()))
    distr = sim_env.objects.get("stove_distr")
    distr_xy = _body_xy(sim_env, distr) if distr is not None else None
    for key in sorted(sites.keys()):
        site = sites[key]
        if site is None or site.get("name") is None:
            continue
        burner_xy = _site_xy(sim_env, site)
        if distr_xy is not None:
            if math.hypot(burner_xy[0] - distr_xy[0], burner_xy[1] - distr_xy[1]
                          ) <= _DISTR_OCCUPY_RADIUS:
                continue
        phrase = _burner_key_to_phrase(key)
        if burner_phrase_to_key(phrase, valid_keys) != key:
            raise RuntimeError(
                f"burner phrase round-trip failed for key {key!r} -> {phrase!r}"
            )
        return phrase
    return None


def _episode_video_path(ep: int) -> str:
    """Per-episode video path from config.VIDEO_PATH parent and extension."""
    base = Path(config.VIDEO_PATH)
    ext = base.suffix if base.suffix else ".mp4"
    return str(base.parent / f"run_ep{ep:03d}{ext}")


def _video_provenance_marker(ep: int) -> str:
    """Video path for logs, or an explicit marker when recording is off."""
    if not config.SAVE_VIDEO:
        return "video_disabled"
    return _episode_video_path(ep)


def _install_grasp_tracker(backend) -> list[int | str]:
    """Wrap backend.step; return one-element list holding first grasp step."""
    grasp_at: list[int | str] = [None]
    orig_step = backend.step

    def step(action: list[float]) -> bool:
        done = orig_step(action)
        if grasp_at[0] is None:
            try:
                sim_env = backend._sim_env()
                gripper = sim_env.robots[0].gripper["right"]
                geoms = sim_env.objects["obj"]
                if sim_env._check_grasp(gripper=gripper, object_geoms=geoms):
                    grasp_at[0] = backend.steps
            except Exception as exc:
                grasp_at[0] = f"unavailable:{type(exc).__name__}"
        return done

    backend.step = step
    return grasp_at


def _run_episode(
    burner_arg: str,
) -> tuple[str, str | None, str, str, bool, bool, bool, int | str]:
    """Return status, burner, place_prompt, task_desc, place/turn/success, grasp."""
    env = SimEnv("kettle_boiling", 0)
    task_desc = env.task_description
    try:
        backend = env._env
        if burner_arg == "auto":
            phrase = _choose_burner_phrase(backend)
            if phrase is None:
                return (
                    "skipped_no_free_burner",
                    None,
                    "n/a",
                    task_desc,
                    False,
                    False,
                    False,
                    "n/a",
                )
        else:
            phrase = burner_arg
        grasp_at = _install_grasp_tracker(backend)
        place_step = {
            "skill": "place",
            "object_name": "kettle",
            "source_name": "counter",
            "destination_name": phrase,
        }
        turn_step = {"skill": "turn_on", "target_name": phrase}
        executor = Executor(env=env)
        place_prompt = executor._build_vla_task(place_step)
        place_res = executor.run([place_step])
        place_ok = place_res.reason == StopReason.PLAN_DONE
        turn_ok = False
        if place_ok:
            turn_res = executor.run([turn_step])
            turn_ok = turn_res.reason == StopReason.PLAN_DONE
        success = env.success()
        grasp = grasp_at[0] if grasp_at[0] is not None else "unavailable"
        return "ok", phrase, place_prompt, task_desc, place_ok, turn_ok, success, grasp
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    place_ok = turn_on_ok = success_ok = skipped = 0
    orig_video_path = config.VIDEO_PATH
    for ep in range(args.episodes):
        video_log = _video_provenance_marker(ep)
        if config.SAVE_VIDEO:
            config.VIDEO_PATH = _episode_video_path(ep)
        try:
            (
                status,
                burner,
                place_prompt,
                task_desc,
                p_ok,
                t_ok,
                s_ok,
                grasp,
            ) = _run_episode(args.burner)
        finally:
            config.VIDEO_PATH = orig_video_path
        if status == "skipped_no_free_burner":
            skipped += 1
            print(
                f"ep={ep} video={video_log!r} task_description={task_desc!r} "
                f"place_prompt={place_prompt!r} burner=None "
                f"status=skipped_no_free_burner place_ok=False turn_on_ok=False "
                f"success=False grasp_step=n/a"
            )
        else:
            place_ok += int(p_ok)
            turn_on_ok += int(t_ok)
            success_ok += int(s_ok)
            print(
                f"ep={ep} video={video_log!r} task_description={task_desc!r} "
                f"place_prompt={place_prompt!r} burner={burner!r} "
                f"place_ok={p_ok} turn_on_ok={t_ok} success={s_ok} "
                f"grasp_step={grasp}"
            )
    n = args.episodes
    print(
        f"summary place_ok {place_ok}/{n} "
        f"turn_on_ok_given_place {turn_on_ok}/{place_ok} "
        f"success {success_ok}/{n} skipped_no_free_burner {skipped}/{n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
