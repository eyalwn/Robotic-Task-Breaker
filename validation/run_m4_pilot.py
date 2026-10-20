"""Run the M4 pilot: bare VLA vs Longor on the drawer open->close chain.

This script runs both arms on ``open_close_drawer`` and writes per-episode
records plus a compact summary under one artifact directory.

Measurement-only script: always exits 0 regardless of episode outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import config  # noqa: E402
from envs.robocasa_env import _OPEN_CLOSE_SUFFIX  # noqa: E402
from envs.sim_env import SimEnv  # noqa: E402
from executor import Executor, StopReason  # noqa: E402
from plan import get_stop_reason, is_stop_plan  # noqa: E402
from planner import LLMPlanner  # noqa: E402
from skills import skill_catalog  # noqa: E402
from utils import configure_logging  # noqa: E402
from vla_client import VLAClient  # noqa: E402

ARM_BARE = "bare"
ARM_LONGOR = "longor"
_ARM_CHOICES = (ARM_BARE, ARM_LONGOR)
_TASK_NAME = "open_close_drawer"

CATEGORY_SUCCESS = "success"
CATEGORY_POLICY_PRIMITIVE = "policy_primitive"
CATEGORY_PLANNER_PLAN = "planner_plan"
CATEGORY_PLAN_TO_SKILL_GROUNDING = "plan_to_skill_grounding"
CATEGORY_SUBTASK_TRANSITION_ORACLE = "subtask_transition_oracle"
CATEGORY_INFRASTRUCTURE = "infrastructure"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--artifact-dir",
                        default=os.environ.get("ARTIFACT_DIR", "artifacts/m4_pilot"))
    parser.add_argument("--arms", default="bare,longor",
                        help="Comma-separated subset of: bare,longor")
    return parser.parse_args()


def _video_path(artifact_dir: Path, arm: str, ep: int) -> str:
    """Build the per-episode video path for one arm."""
    return str(artifact_dir / "videos" / f"{arm}_ep{ep:03d}.mp4")


def _parse_arms(raw: str) -> list[str]:
    """Parse and validate ``--arms`` into execution order."""
    names = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not names:
        raise RuntimeError("at least one arm must be requested")
    invalid = [name for name in names if name not in _ARM_CHOICES]
    if invalid:
        supported = ", ".join(_ARM_CHOICES)
        raise RuntimeError(f"unsupported arm(s) {invalid!r}; supported: {supported}")
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _object_name_from_task_description(task_desc: str) -> str:
    """Parse drawer object name from ``open_close_drawer`` task language."""
    if not task_desc.endswith(_OPEN_CLOSE_SUFFIX):
        raise RuntimeError(
            f"task_description must end with {_OPEN_CLOSE_SUFFIX!r}, "
            f"got {task_desc!r}")
    open_phrase = task_desc[:-len(_OPEN_CLOSE_SUFFIX)].strip()
    prefix = "open the "
    if not open_phrase.lower().startswith(prefix):
        raise RuntimeError(
            f"expected reset language 'open the <name>', got {open_phrase!r}")
    name = open_phrase[len(prefix):].strip()
    if not name:
        raise RuntimeError(f"empty object name in task description {task_desc!r}")
    return name


def _infer_infrastructure(error: str) -> bool:
    """Return whether an error string is likely infrastructure-related."""
    text = error.lower()
    markers = ("vla service", "connection", "timed out", "api key", "503", "404",
               "not installed", "address already in use")
    return any(marker in text for marker in markers)


def _final_longor_status_fields(success: bool, stop_reason: str,
                                error: str) -> tuple[str, str]:
    """Clear stop_reason/error on terminal env success."""
    if success:
        return "", ""
    return stop_reason, error


def _plan_checks(plan_steps: object, expected_object: str) -> tuple[bool, bool]:
    """Return ``(shape_ok, grounding_ok)`` for one planner payload."""
    if not isinstance(plan_steps, list) or not plan_steps:
        return False, False
    open_found = False
    close_found_after_open = False
    all_names_match = True
    for step in plan_steps:
        if not isinstance(step, dict):
            return False, False
        skill = step.get("skill")
        object_name = str(step.get("object_name", "")).strip()
        if skill == "open":
            open_found = True
        if skill == "close" and open_found:
            close_found_after_open = True
        if skill in ("open", "close") and object_name != expected_object:
            all_names_match = False
    shape_ok = open_found and close_found_after_open
    return shape_ok, all_names_match


def _run_bare_episode() -> dict[str, object]:
    """Run one bare-VLA episode with the raw task string as prompt."""
    env = SimEnv(_TASK_NAME, 0)
    task_desc = env.task_description
    object_name = ""
    prompt = task_desc
    done = False
    ever_open = False
    close_now = False
    error = ""
    stop_reason = "episode_timeout"
    try:
        object_name = _object_name_from_task_description(task_desc)
        open_step = {"skill": "open", "object_name": object_name}
        close_step = {"skill": "close", "object_name": object_name}
        client = VLAClient()
        while env.steps < env.horizon and not done:
            images, state = env.observe()
            actions_chunk = client.predict(images, state, prompt)
            for action in actions_chunk:
                done = env.step(action)
                ever_open = ever_open or env.check_subtask(open_step)
                if done or env.steps >= env.horizon:
                    break
        close_now = env.check_subtask(close_step)
        success = env.success()
        stop_reason = "episode_done" if done else "episode_timeout"
        if success:
            category = CATEGORY_SUCCESS
        elif not ever_open or not close_now:
            category = CATEGORY_POLICY_PRIMITIVE
        else:
            category = CATEGORY_SUBTASK_TRANSITION_ORACLE
        return {"success": success, "category": category,
                "task_description": task_desc, "target_object": object_name,
                "prompt": prompt, "stop_reason": stop_reason,
                "steps": env.steps, "error": error}
    except Exception as exc:
        error = str(exc)
        return {"success": False, "category": CATEGORY_INFRASTRUCTURE,
                "task_description": task_desc, "target_object": object_name,
                "prompt": prompt, "stop_reason": "execution_error",
                "steps": env.steps, "error": error}
    finally:
        env.close()


def _run_longor_episode() -> dict[str, object]:
    """Run one Longor episode (LLM planner + executor)."""
    env = None
    task_desc = ""
    target_object = ""
    total_steps = 0
    plan_status = "plan_failed"
    stop_reason = ""
    error = ""
    success = False
    category = CATEGORY_INFRASTRUCTURE
    last_plan: object = None
    shape_ok = True
    grounding_ok = True
    attempts: list[dict[str, object]] = []
    try:
        planner = LLMPlanner(skill_catalog())
        env = SimEnv(_TASK_NAME, 0)
        task_desc = env.task_description
        target_object = _object_name_from_task_description(task_desc)
        executor = Executor(env=env)
        images, _state = env.observe()
        planned = planner.plan(task_desc, images[0])
        if is_stop_plan(planned):
            plan_status = "stopped"
            stop_reason = get_stop_reason(planned)
        else:
            last_plan = planned
            shape_ok, grounding_ok = _plan_checks(planned, target_object)
            plan_steps = list(planned)
            execution_sum = executor.run(plan_steps)
            attempts.append({
                "plan": [dict(step) for step in plan_steps],
                "stop_reason": execution_sum.reason,
                "steps": execution_sum.steps,
                "error": execution_sum.error,
                "next_step_idx": execution_sum.next_step_idx,
                "skill_telemetry": execution_sum.skill_telemetry})
            total_steps = execution_sum.steps
            stop_reason = execution_sum.reason
            error = execution_sum.error
            if execution_sum.reason == StopReason.PLAN_DONE:
                plan_status = "done"
            else:
                plan_status = "execution_error"
        success = bool(env.success())
        if success:
            category = CATEGORY_SUCCESS
        elif plan_status == "stopped":
            category = CATEGORY_PLANNER_PLAN
        elif not shape_ok:
            category = CATEGORY_PLANNER_PLAN
        elif not grounding_ok:
            category = CATEGORY_PLAN_TO_SKILL_GROUNDING
        elif plan_status == "done":
            category = CATEGORY_SUBTASK_TRANSITION_ORACLE
        elif stop_reason in (StopReason.SUBTASK_TIMEOUT, StopReason.EPISODE_TIMEOUT):
            category = CATEGORY_POLICY_PRIMITIVE
        elif stop_reason == StopReason.EXECUTION_ERROR:
            category = (CATEGORY_INFRASTRUCTURE if _infer_infrastructure(error)
                        else CATEGORY_POLICY_PRIMITIVE)
        elif plan_status == "plan_failed":
            category = CATEGORY_PLANNER_PLAN
        else:
            category = CATEGORY_INFRASTRUCTURE
    except Exception as exc:
        error = str(exc)
        plan_status = "plan_failed"
        category = (CATEGORY_INFRASTRUCTURE if _infer_infrastructure(error)
                    else CATEGORY_PLANNER_PLAN)
    finally:
        if env is not None:
            env.close()
    stop_reason, error = _final_longor_status_fields(success, stop_reason, error)
    return {"success": success, "category": category,
            "task_description": task_desc, "target_object": target_object,
            "plan_status": plan_status, "stop_reason": stop_reason,
            "steps": total_steps, "error": error, "last_plan": last_plan,
            "attempts": attempts}


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate outcomes by arm for a compact final report."""
    summary: dict[str, object] = {"arms": {}}
    for arm in _ARM_CHOICES:
        arm_rows = [row for row in records if row["arm"] == arm]
        if not arm_rows:
            continue
        success_count = sum(1 for row in arm_rows if bool(row["success"]))
        counts = Counter(str(row["category"]) for row in arm_rows)
        summary["arms"][arm] = {
            "episodes": len(arm_rows), "successes": success_count,
            "success_rate": (success_count / len(arm_rows)) if arm_rows else 0.0,
            "category_counts": dict(counts)}
    return summary


def main() -> int:
    configure_logging()
    args = _parse_args()
    arms = _parse_arms(args.arms)
    artifact_dir = Path(args.artifact_dir)
    videos_dir = artifact_dir / "videos"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / "m4_pilot_episodes.jsonl"
    summary_path = artifact_dir / "m4_pilot_summary.json"

    records: list[dict[str, object]] = []
    original_video_path = config.VIDEO_PATH
    try:
        for ep in range(args.episodes):
            for arm in arms:
                if config.SAVE_VIDEO:
                    config.VIDEO_PATH = _video_path(artifact_dir, arm, ep)
                    video_marker = config.VIDEO_PATH
                else:
                    video_marker = "video_disabled"
                started = time.perf_counter()
                run = _run_bare_episode if arm == ARM_BARE else _run_longor_episode
                row = run()
                elapsed = time.perf_counter() - started
                row["arm"] = arm
                row["episode"] = ep
                row["video"] = video_marker
                row["elapsed_s"] = round(elapsed, 3)
                print(f"arm={arm} ep={ep} video={video_marker!r} "
                      f"task_description={row['task_description']!r} "
                      f"success={row['success']} category={row['category']} "
                      f"steps={row['steps']} "
                      f"stop_reason={row.get('stop_reason', '')!r} "
                      f"error={row['error']!r}")
                records.append(row)
    finally:
        config.VIDEO_PATH = original_video_path

    with records_path.open("w", encoding="utf-8") as fp:
        for row in records:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    summary = _summarize(records)
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(summary, indent=2) + "\n")

    print(f"episodes_file={records_path}")
    print(f"summary_file={summary_path}")
    for arm, arm_sum in summary["arms"].items():
        print(f"summary arm={arm} "
              f"success={arm_sum['successes']}/{arm_sum['episodes']} "
              f"rate={arm_sum['success_rate']:.3f} "
              f"categories={arm_sum['category_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
