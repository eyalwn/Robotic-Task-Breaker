"""Run the M4 pilot: bare VLA vs Longor on the drawer open->close chain.

Paired protocol only: one scene per pair, backend checkpoint restore before
each arm, counterbalanced order, fail-closed observability matching.

Writes per-episode records and a compact summary under one artifact directory.
Exits non-zero when a pair fails observability matching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
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
_TASK_NAME = "open_close_drawer"

CATEGORY_SUCCESS = "success"
CATEGORY_POLICY_PRIMITIVE = "policy_primitive"
CATEGORY_PLANNER_PLAN = "planner_plan"
CATEGORY_PLAN_TO_SKILL_GROUNDING = "plan_to_skill_grounding"
CATEGORY_SUBTASK_TRANSITION_ORACLE = "subtask_transition_oracle"
CATEGORY_INFRASTRUCTURE = "infrastructure"


_VLA_HEALTH_URL = "http://127.0.0.1:8765/health"


def _pid_alive(pid: int) -> bool:
    """Return whether ``pid`` is a live process on this host."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(code)):
                return False
            return int(code.value) == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _fetch_vla_health(url: str, timeout_s: float) -> tuple[bool, str]:
    """GET health URL; return ``(ok, body_or_error)``."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return True, body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def run_vla_preflight(
        artifact_dir: Path,
        adapter_pid: int | None = None,
        health_url: str = _VLA_HEALTH_URL,
        timeout_s: float = 5.0) -> dict[str, object]:
    """Verify VLA adapter PID (if given) and health before a scored M4 batch."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pid_path = artifact_dir / "adapter.pid"
    if adapter_pid is None and pid_path.is_file():
        text = pid_path.read_text(encoding="utf-8").strip()
        if text.isdigit():
            adapter_pid = int(text)
    health_ok, health_body = _fetch_vla_health(health_url, timeout_s)
    pid_ok: bool | None = None
    if adapter_pid is not None:
        pid_ok = _pid_alive(adapter_pid)
    payload: dict[str, object] = {
        "health_url": health_url,
        "health_ok": health_ok,
        "health_body": health_body,
        "adapter_pid": adapter_pid,
        "adapter_pid_alive": pid_ok,
    }
    out_path = artifact_dir / "vla_preflight.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                        encoding="utf-8")
    failures: list[str] = []
    if adapter_pid is not None and pid_ok is False:
        failures.append(f"adapter PID {adapter_pid} is not running")
    if not health_ok:
        failures.append(f"health check failed for {health_url}: {health_body}")
    payload["ok"] = not failures
    if failures:
        payload["error"] = "; ".join(failures)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                            encoding="utf-8")
        raise RuntimeError(payload["error"])
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                        encoding="utf-8")
    return payload


class PairObservabilityMismatch(RuntimeError):
    """Raised when paired arms do not share identical initial observables."""

    def __init__(self, reason: str, rows: list[dict[str, object]]):
        super().__init__(reason)
        self.rows = rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of paired episodes (one scene per pair)")
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("ARTIFACT_DIR", "artifacts/m4_pilot"))
    return parser.parse_args()


def _video_path(artifact_dir: Path, arm: str, pair_id: int) -> str:
    """Build the per-pair video path for one arm."""
    return str(artifact_dir / "videos" / f"{arm}_ep{pair_id:03d}.mp4")


def _arm_order_for_pair(pair_id: int) -> tuple[str, str]:
    """Counterbalance execution order by pair index."""
    if pair_id % 2 == 0:
        return (ARM_BARE, ARM_LONGOR)
    return (ARM_LONGOR, ARM_BARE)


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


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _state_vector_hash(state: list[float]) -> str:
    parts = [f"{float(v):.17g}" for v in state]
    return _sha256_hex(",".join(parts).encode("utf-8"))


def _json_stable_hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_hex(text.encode("utf-8"))


def _fingerprint(env: SimEnv) -> dict[str, str | list[str]]:
    """Hash initial observables before policy inference."""
    images, state = env.observe()
    task_desc = env.task_description
    target = _object_name_from_task_description(task_desc)
    return {
        "task_description": _sha256_hex(task_desc.encode("utf-8")),
        "target_object": _sha256_hex(target.encode("utf-8")),
        "policy_png": [_sha256_hex(b) for b in images],
        "state_vector": _state_vector_hash(list(state)),
        "oracle_state": _json_stable_hash(env.drawer_joint_state()),
        "wrapper_state": _json_stable_hash(env.episode_wrapper_state()),
    }


def _hash_mismatch_reason(baseline: dict[str, str | list[str]],
                          current: dict[str, str | list[str]]) -> str:
    for key in sorted(baseline):
        if baseline[key] != current[key]:
            return f"initial observability mismatch on '{key}'"
    return ""


def _infer_infrastructure(error: str) -> bool:
    """Return whether an error string is likely infrastructure-related."""
    text = error.lower()
    markers = ("vla service", "connection", "timed out", "api key", "503", "404",
               "not installed", "address already in use")
    return any(marker in text for marker in markers)


def _longor_category(success: bool, plan_status: str, shape_ok: bool,
                     grounding_ok: bool, stop_reason: str,
                     error: str) -> str:
    """Classify a Longor episode outcome (same branch order as legacy inline)."""
    if success:
        return CATEGORY_SUCCESS
    if plan_status == "stopped":
        return CATEGORY_PLANNER_PLAN
    if not shape_ok:
        return CATEGORY_PLANNER_PLAN
    if not grounding_ok:
        return CATEGORY_PLAN_TO_SKILL_GROUNDING
    if plan_status == "done":
        return CATEGORY_SUBTASK_TRANSITION_ORACLE
    if stop_reason in (StopReason.SUBTASK_TIMEOUT, StopReason.EPISODE_TIMEOUT):
        return CATEGORY_POLICY_PRIMITIVE
    if stop_reason == StopReason.EXECUTION_ERROR:
        return (CATEGORY_INFRASTRUCTURE if _infer_infrastructure(error)
                else CATEGORY_POLICY_PRIMITIVE)
    if plan_status == "plan_failed":
        return CATEGORY_PLANNER_PLAN
    return CATEGORY_INFRASTRUCTURE


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


def _run_bare_episode(env: SimEnv) -> dict[str, object]:
    """Run one bare-VLA episode with the raw task string as prompt."""
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


def _run_longor_episode(env: SimEnv) -> dict[str, object]:
    """Run one Longor episode (LLM planner + executor)."""
    task_desc = env.task_description
    target_object = ""
    total_steps = 0
    plan_status = "plan_failed"
    stop_reason = ""
    error = ""
    success = False
    last_plan: object = None
    shape_ok = True
    grounding_ok = True
    attempts: list[dict[str, object]] = []
    try:
        planner = LLMPlanner(skill_catalog())
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
        category = _longor_category(
            success, plan_status, shape_ok, grounding_ok, stop_reason, error)
    except Exception as exc:
        error = str(exc)
        plan_status = "plan_failed"
        category = (CATEGORY_INFRASTRUCTURE if _infer_infrastructure(error)
                    else CATEGORY_PLANNER_PLAN)
    stop_reason, error = _final_longor_status_fields(success, stop_reason, error)
    return {"success": success, "category": category,
            "task_description": task_desc, "target_object": target_object,
            "plan_status": plan_status, "stop_reason": stop_reason,
            "steps": total_steps, "error": error, "last_plan": last_plan,
            "attempts": attempts}


def _invalid_arm_row(arm: str, pair_id: int, arm_order: str,
                     reason: str) -> dict[str, object]:
    return {
        "arm": arm, "pair_id": pair_id, "arm_order": arm_order,
        "pair_valid": False, "pair_mismatch_reason": reason,
        "success": False, "category": CATEGORY_INFRASTRUCTURE,
        "task_description": "", "target_object": "",
        "stop_reason": "pair_invalid", "steps": 0,
        "error": reason, "initial_hashes": {},
    }


def _mismatch_rows(first_row: dict[str, object], second_arm: str, pair_id: int,
                   order_str: str, reason: str) -> list[dict[str, object]]:
    """Mark the first arm invalid and append a stub row for the second arm."""
    first_row["pair_valid"] = False
    first_row["pair_mismatch_reason"] = reason
    return [first_row, _invalid_arm_row(second_arm, pair_id, order_str, reason)]


def _require_matching_hashes(
        baseline: dict[str, str | list[str]],
        hashes: dict[str, str | list[str]],
        first_row: dict[str, object], second_arm: str, pair_id: int,
        order_str: str) -> None:
    """Fail closed when the second arm's initial observables differ."""
    reason = _hash_mismatch_reason(baseline, hashes)
    if not reason:
        return
    rows = _mismatch_rows(first_row, second_arm, pair_id, order_str, reason)
    raise PairObservabilityMismatch(reason, rows)


def _run_arm(env: SimEnv, arm: str, pair_id: int, order_str: str,
             hashes: dict[str, str | list[str]], artifact_dir: Path,
             want_video: bool) -> dict[str, object]:
    """Configure video, run one arm, return a completed episode row."""
    if want_video:
        config.SAVE_VIDEO = True
        config.VIDEO_PATH = _video_path(artifact_dir, arm, pair_id)
        video_marker = config.VIDEO_PATH
    else:
        config.SAVE_VIDEO = False
        video_marker = "video_disabled"
    started = time.perf_counter()
    runner = _run_bare_episode if arm == ARM_BARE else _run_longor_episode
    row = runner(env)
    elapsed = time.perf_counter() - started
    row["arm"] = arm
    row["pair_id"] = pair_id
    row["arm_order"] = order_str
    row["pair_valid"] = True
    row["pair_mismatch_reason"] = ""
    row["initial_hashes"] = hashes
    row["video"] = video_marker
    row["elapsed_s"] = round(elapsed, 3)
    print(f"pair={pair_id} arm={arm} order={order_str!r} "
          f"video={video_marker!r} "
          f"task_description={row['task_description']!r} "
          f"success={row['success']} category={row['category']} "
          f"steps={row['steps']} "
          f"stop_reason={row.get('stop_reason', '')!r} "
          f"error={row['error']!r}")
    return row


def _run_pair(pair_id: int, artifact_dir: Path,
              want_video: bool) -> list[dict[str, object]]:
    """One SimEnv per pair; restore checkpoint before each arm."""
    arm_order = _arm_order_for_pair(pair_id)
    order_str = ",".join(arm_order)
    config.SAVE_VIDEO = False
    env = SimEnv(_TASK_NAME, 0)
    try:
        checkpoint = env.capture_episode_checkpoint()
        baseline: dict[str, str | list[str]] | None = None
        pair_rows: list[dict[str, object]] = []
        for arm in arm_order:
            config.SAVE_VIDEO = False
            env.restore_episode_checkpoint(checkpoint)
            hashes = _fingerprint(env)
            if baseline is None:
                baseline = hashes
            else:
                _require_matching_hashes(
                    baseline, hashes, pair_rows[0], arm, pair_id, order_str)
            pair_rows.append(_run_arm(
                env, arm, pair_id, order_str, hashes, artifact_dir, want_video))
        return pair_rows
    finally:
        env.close()


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate outcomes by arm for a compact final report."""
    valid = [row for row in records if bool(row.get("pair_valid", False))]
    summary: dict[str, object] = {
        "pairs_total": len({row["pair_id"] for row in records}),
        "pairs_valid": len({row["pair_id"] for row in valid}),
        "arms": {},
    }
    for arm in (ARM_BARE, ARM_LONGOR):
        arm_rows = [row for row in valid if row["arm"] == arm]
        if not arm_rows:
            continue
        success_count = sum(1 for row in arm_rows if bool(row["success"]))
        counts = Counter(str(row["category"]) for row in arm_rows)
        summary["arms"][arm] = {
            "episodes": len(arm_rows), "successes": success_count,
            "success_rate": (success_count / len(arm_rows)) if arm_rows else 0.0,
            "category_counts": dict(counts)}
    return summary


def _write_artifacts(records_path: Path, summary_path: Path,
                     records: list[dict[str, object]]) -> dict[str, object]:
    with records_path.open("w", encoding="utf-8") as fp:
        for row in records:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    summary = _summarize(records)
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    configure_logging()
    args = _parse_args()
    artifact_dir = Path(args.artifact_dir)
    videos_dir = artifact_dir / "videos"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / "m4_pilot_episodes.jsonl"
    summary_path = artifact_dir / "m4_pilot_summary.json"

    records: list[dict[str, object]] = []
    original_save_video = config.SAVE_VIDEO
    original_video_path = config.VIDEO_PATH
    exit_code = 0
    try:
        for pair_id in range(args.episodes):
            records.extend(_run_pair(pair_id, artifact_dir, original_save_video))
    except PairObservabilityMismatch as exc:
        records.extend(exc.rows)
        _write_artifacts(records_path, summary_path, records)
        print(f"pair_invalid: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        config.SAVE_VIDEO = original_save_video
        config.VIDEO_PATH = original_video_path

    summary = _write_artifacts(records_path, summary_path, records)
    print(f"episodes_file={records_path}")
    print(f"summary_file={summary_path}")
    for arm, arm_sum in summary["arms"].items():
        print(f"summary arm={arm} "
              f"success={arm_sum['successes']}/{arm_sum['episodes']} "
              f"rate={arm_sum['success_rate']:.3f} "
              f"categories={arm_sum['category_counts']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
