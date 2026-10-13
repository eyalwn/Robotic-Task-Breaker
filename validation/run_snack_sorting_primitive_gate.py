"""M3 SnackSorting primitive gate: fixed two-step plan, no LLM.

Runs put_in then close inside the SnackSorting scene with per-stage oracle
checks. Colab: after ``run_longor_task.ipynb`` section 3, source
``.colab_m2_env`` and point ``ROBO_PY`` at this script (VLA HTTP on :8765).

Measurement-only script: always exits 0 regardless of episode outcomes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import config  # noqa: E402
from envs.sim_env import SimEnv  # noqa: E402
from executor import Executor, StopReason  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    return parser.parse_args()


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


def _run_episode() -> tuple[str, str, str, bool, bool, bool]:
    """Return task_desc, put_in_prompt, close_prompt, put_in_ok, close_ok, success."""
    env = SimEnv("snack_sorting", 0)
    task_desc = env.task_description
    try:
        put_step = {
            "skill": "put_in",
            "object_name": "bar",
            "source_name": "drawer",
            "container_name": "bowl",
        }
        close_step = {"skill": "close", "object_name": "drawer"}
        executor = Executor(env=env)
        put_prompt = executor._build_vla_task(put_step)
        close_prompt = executor._build_vla_task(close_step)
        put_res = executor.run([put_step])
        put_ok = put_res.reason == StopReason.PLAN_DONE
        close_ok = False
        if put_ok:
            close_res = executor.run([close_step])
            close_ok = close_res.reason == StopReason.PLAN_DONE
        success = env.success()
        return task_desc, put_prompt, close_prompt, put_ok, close_ok, success
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    put_in_ok = close_ok = success_ok = 0
    orig_video_path = config.VIDEO_PATH
    for ep in range(args.episodes):
        video_log = _video_provenance_marker(ep)
        if config.SAVE_VIDEO:
            config.VIDEO_PATH = _episode_video_path(ep)
        try:
            task_desc, put_prompt, close_prompt, p_ok, c_ok, s_ok = _run_episode()
        finally:
            config.VIDEO_PATH = orig_video_path
        put_in_ok += int(p_ok)
        close_ok += int(c_ok)
        success_ok += int(s_ok)
        print(
            f"ep={ep} video={video_log!r} task_description={task_desc!r} "
            f"put_in_prompt={put_prompt!r} close_prompt={close_prompt!r} "
            f"put_in_ok={p_ok} close_ok={c_ok} success={s_ok}"
        )
    n = args.episodes
    print(
        f"summary put_in_ok {put_in_ok}/{n} "
        f"close_ok_given_put_in {close_ok}/{put_in_ok} success {success_ok}/{n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
