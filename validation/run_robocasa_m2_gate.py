"""Run the RoboCasa M2 gate (adapter parity + Longor vertical slice)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_VALIDATION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VALIDATION_DIR.parent
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_SRC_ROOT, _VALIDATION_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import config

from robocasa_m2_verdict import (
    build_verdict,
    parse_longor_runner_stdout,
    parse_service_raw_stdout,
    stage_a_blocked,
    stage_a_from_service_successes,
    stage_b_blocked,
    stage_b_from_runner_success,
    upstream_gate_verdict_line,
    upstream_reference_from_rollout_log,
    utc_run_id,
    write_verdict_artifacts,
    StageVerdict,
)

_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "validation" / "artifacts" / "robocasa_m2_gate"
_RAW_POLICY = _REPO_ROOT / "examples" / "run_robocasa_raw_policy.py"
_LONGOR_TASK = _REPO_ROOT / "examples" / "run_longor_task.py"


def _vla_health_ok() -> tuple[bool, str]:
    """Return whether the Longor VLA HTTP service responds on /health."""
    url = f"{config.VLA_URL}/health"
    try:
        with urllib.request.urlopen(url, timeout=config.VLA_TIMEOUT) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return False, f"VLA /health unreachable at {url}: {exc}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False, f"VLA /health returned non-JSON body from {url}"
    if payload.get("status") != "ok":
        return False, f"VLA /health status not ok: {payload!r}"
    return True, ""


def _linux_robocasa_ready() -> tuple[bool, str]:
    """Return whether this host can import the RoboCasa sim backend."""
    if sys.platform == "win32":
        return False, "RoboCasa sim backend is Linux-only (Windows host)"
    try:
        from envs.robocasa_env import RoboCasaEnv  # noqa: F401
    except ImportError as exc:
        return False, f"RoboCasa backend unavailable: {exc}"
    return True, ""


def _run_subprocess(script: Path, args: list[str]) -> tuple[int, str]:
    """Run a repo example script and capture combined stdout/stderr text."""
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    text = proc.stdout
    if proc.stderr:
        text = f"{text}\n{proc.stderr}" if text else proc.stderr
    return proc.returncode, text


def _run_stage_a_live() -> tuple[StageVerdict, str]:
    """Execute service-backed raw OpenDrawer episodes when prerequisites hold."""
    if config.SIM_ENV != "robocasa":
        return stage_a_blocked(
            f"config.SIM_ENV must be 'robocasa' for stage A (got {config.SIM_ENV!r})"
        ), ""
    if config.VLA_POLICY != "groot":
        return stage_a_blocked(
            "config.VLA_POLICY must be 'groot' for stage A adapter parity "
            f"(got {config.VLA_POLICY!r})"
        ), ""
    sim_ok, sim_reason = _linux_robocasa_ready()
    if not sim_ok:
        return stage_a_blocked(sim_reason), ""
    health_ok, health_reason = _vla_health_ok()
    if not health_ok:
        return stage_a_blocked(health_reason), ""
    exit_code, text = _run_subprocess(_RAW_POLICY, [])
    if exit_code != 0:
        return stage_a_blocked(
            f"run_robocasa_raw_policy.py exited {exit_code}; see stage_a_service_raw.log"
        ), text
    successes, episodes, note = parse_service_raw_stdout(text)
    if successes is None or episodes is None:
        return stage_a_blocked(f"could not parse service raw stdout: {note}"), text
    return stage_a_from_service_successes(successes, episodes), text


def _run_stage_b_live() -> tuple[StageVerdict, str]:
    """Run one frozen OpenDrawer Longor episode when prerequisites hold."""
    if config.SIM_ENV != "robocasa":
        return stage_b_blocked(
            f"config.SIM_ENV must be 'robocasa' for stage B (got {config.SIM_ENV!r})"
        ), ""
    if config.VLA_POLICY != "groot":
        return stage_b_blocked(
            "config.VLA_POLICY must be 'groot' for stage B "
            f"(got {config.VLA_POLICY!r})"
        ), ""
    sim_ok, sim_reason = _linux_robocasa_ready()
    if not sim_ok:
        return stage_b_blocked(sim_reason), ""
    health_ok, health_reason = _vla_health_ok()
    if not health_ok:
        return stage_b_blocked(health_reason), ""
    exit_code, text = _run_subprocess(
        _LONGOR_TASK,
        ["--task", "open_drawer", "--task-idx", "0"],
    )
    if exit_code not in (0, 1):
        return stage_b_blocked(
            f"run_longor_task.py exited {exit_code}; see stage_b_longor_run.log"
        ), text
    parsed, note = parse_longor_runner_stdout(text)
    if parsed is None:
        return stage_b_blocked(f"could not parse runner stdout: {note}"), text
    return stage_b_from_runner_success(parsed), text


def main() -> int:
    """Parse CLI flags, run live stages when allowed, and write verdict artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help="Parent directory for timestamped M2 gate artifact folders",
    )
    parser.add_argument(
        "--run-utc",
        default="",
        help="UTC timestamp folder name (default: now)",
    )
    parser.add_argument(
        "--upstream-rollout-log",
        type=Path,
        default=None,
        help="Optional M1 upstream rollout.log for reference metadata only",
    )
    parser.add_argument(
        "--upstream-verdict-txt",
        type=Path,
        default=None,
        help="Optional M1-style verdict.txt path (default: rollout.log parent when name is rollout.log)",
    )
    parser.add_argument(
        "--run-live-stages",
        action="store_true",
        help="Attempt live stage A/B (Linux RoboCasa + GR00T VLA service required)",
    )
    args = parser.parse_args()

    run_utc = args.run_utc.strip() or utc_run_id(datetime.now(timezone.utc))
    artifact_dir = args.output_root / run_utc

    upstream_ref = None
    if args.upstream_rollout_log is not None:
        upstream_ref = upstream_reference_from_rollout_log(args.upstream_rollout_log)
        upstream_line = upstream_gate_verdict_line(upstream_ref)
        print(upstream_line)

    if args.run_live_stages:
        stage_a, stage_a_log = _run_stage_a_live()
        stage_b, stage_b_log = _run_stage_b_live()
    else:
        stage_a = stage_a_blocked(
            "live stage A not requested; pass --run-live-stages on Linux with "
            "RoboCasa + GR00T VLA service"
        )
        stage_a_log = ""
        stage_b = stage_b_blocked(
            "live stage B not requested; pass --run-live-stages on Linux with "
            "RoboCasa + GR00T VLA service"
        )
        stage_b_log = ""

    verdict = build_verdict(run_utc, stage_a, stage_b, upstream_ref)
    write_verdict_artifacts(artifact_dir, verdict)

    if args.upstream_rollout_log is not None and upstream_ref is not None:
        upstream_line = upstream_gate_verdict_line(upstream_ref)
        (artifact_dir / "upstream_primitive_verdict.txt").write_text(
            upstream_line + "\n", encoding="utf-8"
        )
        verdict_txt = args.upstream_verdict_txt
        if verdict_txt is None and args.upstream_rollout_log.name == "rollout.log":
            verdict_txt = args.upstream_rollout_log.parent / "verdict.txt"
        if verdict_txt is not None:
            verdict_txt.parent.mkdir(parents=True, exist_ok=True)
            verdict_txt.write_text(upstream_line + "\n", encoding="utf-8")

    if stage_a_log:
        (artifact_dir / "stage_a_service_raw.log").write_text(
            stage_a_log, encoding="utf-8"
        )
    if stage_b_log:
        (artifact_dir / "stage_b_longor_run.log").write_text(
            stage_b_log, encoding="utf-8"
        )

    print(json.dumps(verdict.to_dict(), indent=2))
    print(f"artifact_dir={artifact_dir}")
    return 0 if verdict.overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
