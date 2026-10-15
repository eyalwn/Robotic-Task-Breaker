"""M3 drawer open->close chain gate: fixed two-step plan, no LLM.

Runs open then close inside the OpenDrawer scene (``open_close_drawer`` task)
with per-stage oracle checks. Colab: after ``run_longor_task.ipynb`` section 3,
source ``.colab_m2_env`` and point ``ROBO_PY`` at this script (VLA HTTP on :8765).

Measurement-only script: always exits 0 regardless of episode outcomes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import config  # noqa: E402
from envs.robocasa_env import _OPEN_CLOSE_SUFFIX  # noqa: E402
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


def _object_name_from_task_description(task_desc: str) -> str:
    """Parse drawer object name from ``open_close_drawer`` task description."""
    if not task_desc.endswith(_OPEN_CLOSE_SUFFIX):
        raise RuntimeError(
            f"task_description must end with {_OPEN_CLOSE_SUFFIX!r}, "
            f"got {task_desc!r}"
        )
    open_phrase = task_desc[: -len(_OPEN_CLOSE_SUFFIX)].strip()
    prefix = "open the "
    if not open_phrase.lower().startswith(prefix):
        raise RuntimeError(
            f"expected reset language 'open the <name>', got {open_phrase!r}"
        )
    name = open_phrase[len(prefix) :].strip()
    if not name:
        raise RuntimeError(
            f"empty object name in task description {task_desc!r}")
    return name


def _run_episode() -> tuple[str, str, str, bool, bool, bool]:
    """Return task_desc, open_prompt, close_prompt, open_ok, close_ok, success."""
    env = SimEnv("open_close_drawer", 0)
    task_desc = env.task_description
    try:
        object_name = _object_name_from_task_description(task_desc)
        open_step = {"skill": "open", "object_name": object_name}
        close_step = {"skill": "close", "object_name": object_name}
        executor = Executor(env=env)
        open_prompt = executor._build_vla_task(open_step)
        close_prompt = executor._build_vla_task(close_step)
        open_res = executor.run([open_step])
        open_ok = open_res.reason == StopReason.PLAN_DONE
        close_ok = False
        if open_ok:
            close_res = executor.run([close_step])
            close_ok = close_res.reason == StopReason.PLAN_DONE
        success = env.success()
        return task_desc, open_prompt, close_prompt, open_ok, close_ok, success
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    open_ok = close_ok = success_ok = 0
    orig_video_path = config.VIDEO_PATH
    for ep in range(args.episodes):
        video_log = _video_provenance_marker(ep)
        if config.SAVE_VIDEO:
            config.VIDEO_PATH = _episode_video_path(ep)
        try:
            task_desc, open_prompt, close_prompt, o_ok, c_ok, s_ok = (
                _run_episode())
        finally:
            config.VIDEO_PATH = orig_video_path
        open_ok += int(o_ok)
        close_ok += int(c_ok)
        success_ok += int(s_ok)
        print(
            f"ep={ep} video={video_log!r} task_description={task_desc!r} "
            f"open_prompt={open_prompt!r} close_prompt={close_prompt!r} "
            f"open_ok={o_ok} close_ok={c_ok} success={s_ok}"
        )
    n = args.episodes
    print(
        f"summary open_ok {open_ok}/{n} "
        f"close_ok_given_open {close_ok}/{open_ok} success {success_ok}/{n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
