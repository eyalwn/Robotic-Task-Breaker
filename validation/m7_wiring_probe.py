"""M7.2 sim-GT isolated VLA→arm wiring probe (no planner/executor/LLM)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "m7"
DEFAULT_GATE = "m7.2"
DEFAULT_SUITE = "libero_object"
DEFAULT_TASK_IDX = 0
DEFAULT_STEPS = 8
DEFAULT_MIN_DISPLACEMENT = 0.001
DEFAULT_TASK_PROMPT = "pick up the object"

_SECRET_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PAT", "CREDENTIAL")
_ENV_SNAPSHOT_KEYS = (
    "BACKEND",
    "GITHUB_BRANCH",
    "LIBERO_CONTROL_MODE",
    "LIBERO_EE_DELTA_SCALE",
    "LIBERO_OBS_TYPE",
    "LIBERO_OBSERVATION_HEIGHT",
    "LIBERO_OBSERVATION_WIDTH",
    "LIBERO_SETTLE_STEPS",
    "LIBERO_VLA_IMAGE_FLIP",
    "LOG_LEVEL",
    "M7_PROBE_TASK",
    "MUJOCO_GL",
    "PERCEPTION_BACKEND",
    "PYOPENGL_PLATFORM",
    "ROBOCREW_SIM_SEED",
    "VLA_CHUNK_STEPS",
    "VLA_GRIPPER_MODE",
    "VLA_POLICY",
    "VLA_PORT",
    "VLA_STEPS_PER_ACTION",
)

logger = logging.getLogger(__name__)


def eef_displacement(
    before: list[float] | None,
    after: list[float] | None,
) -> float | None:
    """Return L2 norm of EE motion, or None when either pose is missing."""
    if before is None or after is None:
        return None
    if len(before) < 3 or len(after) < 3:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(before[:3], after[:3])))


def evaluate_m7_2(
    *,
    completed_without_error: bool,
    displacement: float | None,
    min_displacement: float,
) -> tuple[str, str]:
    """Return (status, reason) for the M7.2 gate."""
    if not completed_without_error:
        return "fail", "probe raised an exception"
    if displacement is None:
        return "fail", "missing EE position before/after probe"
    if displacement < min_displacement:
        return (
            "fail",
            f"EE displacement {displacement:.6f}m below threshold {min_displacement:.6f}m",
        )
    return "pass", f"EE displaced {displacement:.6f}m (>= {min_displacement:.6f}m)"


def _is_secret_env_key(key: str) -> bool:
    upper = key.upper()
    return any(part in upper for part in _SECRET_SUBSTRINGS)


def collect_env_snapshot(env: dict[str, str] | None = None) -> dict[str, str]:
    """Collect non-secret env keys for artifact summary."""
    source = env if env is not None else os.environ
    snapshot: dict[str, str] = {}
    for key in _ENV_SNAPSHOT_KEYS:
        value = source.get(key)
        if value is not None and not _is_secret_env_key(key):
            snapshot[key] = value
    return snapshot


def git_metadata() -> tuple[str, str]:
    """Return (branch, sha) best-effort from git."""
    branch = "unknown"
    sha = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return branch, sha


def ensure_libero_config() -> str:
    """Create ~/.libero/config.yaml when missing (headless-safe)."""
    config_root = Path(
        os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))
    )
    config_path = config_root / "config.yaml"
    if config_path.exists():
        return str(config_path)

    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise RuntimeError("libero package not found while preparing LIBERO config")

    libero_pkg = Path(spec.origin).resolve().parent / "libero"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"benchmark_root: {libero_pkg.as_posix()}",
                f"bddl_files: {(libero_pkg / 'bddl_files').as_posix()}",
                f"init_states: {(libero_pkg / 'init_files').as_posix()}",
                f"datasets: {(libero_pkg.parent / 'datasets').as_posix()}",
                f"assets: {(libero_pkg / 'assets').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Created LIBERO config file at %s", config_path)
    return str(config_path)


def write_summary(run_dir: Path, summary: dict[str, Any]) -> Path:
    """Write summary.json under the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _apply_action(arm: Any, cfg: Any, raw_action: list[float]) -> None:
    from vla.action import process_raw_action

    if len(raw_action) == 7:
        clipped = [max(-cfg.action_clip, min(cfg.action_clip, v)) for v in raw_action]
        arm.apply_vla_action(
            clipped[0],
            clipped[1],
            clipped[2],
            clipped[3],
            clipped[4],
            clipped[5],
            clipped[6],
            steps=cfg.steps_per_action,
        )
        return
    if len(raw_action) == 4:
        dx, dy, dz = process_raw_action(raw_action[:3], cfg)
        arm.move_ee(dx, dy, dz, steps=cfg.steps_per_action)
        return
    raise ValueError(f"VLA action must have 4 or 7 elements, got {len(raw_action)}")


def _close_env(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.warning("env.close() failed: %s", exc)


def run_wiring_probe(
    *,
    task_suite: str,
    task_idx: int,
    steps: int,
    task_prompt: str,
    server_url: str,
    seed: int | None,
) -> dict[str, Any]:
    """Run one bounded predict/apply loop and return metrics."""
    from sim.libero_backend import LiberoArm, LiberoCamera, create_env
    from sim.libero_smoke import apply_smoke_env_defaults
    from vla.client import VLAClient, VLAServiceError
    from vla.config import VLAConfig

    apply_smoke_env_defaults()
    if seed is not None:
        os.environ["ROBOCREW_SIM_SEED"] = str(seed)
    ensure_libero_config()

    cfg = VLAConfig.from_env()
    client = VLAClient(server_url, timeout=cfg.request_timeout)
    client.health_check()

    env = create_env(task_suite=task_suite, task_idx=task_idx)
    arm = LiberoArm(env)
    camera = LiberoCamera(env, arm=arm)

    ee_before = list(arm.last_tcp_pos) if arm.last_tcp_pos else None
    steps_applied = 0
    last_error: str | None = None

    try:
        chunk_limit = max(1, int(cfg.chunk_steps or 1))
        while steps_applied < steps:
            if getattr(arm, "episode_done", False):
                logger.warning("Episode ended early at step %d", steps_applied)
                break

            image_bytes = camera.capture_image()
            wrist_image_bytes = None
            capture_wrist = getattr(camera, "capture_wrist_image", None)
            if callable(capture_wrist):
                wrist_image_bytes = capture_wrist()

            arm_state = getattr(arm, "agent_state", None)
            state_list = arm_state.tolist() if arm_state is not None else None
            chunk = client.predict(
                image_bytes,
                task_prompt,
                state=state_list,
                wrist_image_bytes=wrist_image_bytes,
            )

            for raw_action in chunk[:chunk_limit]:
                if steps_applied >= steps:
                    break
                if getattr(arm, "episode_done", False):
                    break
                _apply_action(arm, cfg, list(raw_action))
                steps_applied += 1
    except VLAServiceError as exc:
        last_error = str(exc)
        logger.error("VLA service error: %s", exc)
    finally:
        ee_after = list(arm.last_tcp_pos) if arm.last_tcp_pos else None
        _close_env(env)

    displacement = eef_displacement(ee_before, ee_after)
    return {
        "ee_before": ee_before,
        "ee_after": ee_after,
        "displacement_m": displacement,
        "steps_applied": steps_applied,
        "seed": seed,
        "task_prompt": task_prompt,
        "error": last_error,
        "completed_without_error": last_error is None,
    }


def _drift_between_runs(first: dict[str, Any], second: dict[str, Any]) -> float | None:
    return eef_displacement(first.get("ee_after"), second.get("ee_after"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M7 wiring probe (isolated VLA→arm).")
    parser.add_argument("--gate", default=DEFAULT_GATE, help="M7 gate id (default: m7.2)")
    parser.add_argument("--task-suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-idx", type=int, default=DEFAULT_TASK_IDX)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=DEFAULT_MIN_DISPLACEMENT,
        help="Minimum EE displacement (m) required to pass",
    )
    parser.add_argument(
        "--task-prompt",
        default=os.environ.get("M7_PROBE_TASK", DEFAULT_TASK_PROMPT),
    )
    parser.add_argument(
        "--vla-url",
        default=os.environ.get(
            "VLA_URL",
            f"http://localhost:{os.environ.get('VLA_PORT', '8765')}",
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="ROBOCREW_SIM_SEED override (default: env or 0)",
    )
    parser.add_argument(
        "--drift-check",
        action="store_true",
        help="Run two short probes with the same seed and report EE drift",
    )
    return parser.parse_args(argv)


def _configure_logging(run_log: Path) -> None:
    from sim.libero_smoke import attach_run_log_file, configure_smoke_logging

    os.environ["LONGOR_RUN_LOG"] = str(run_log)
    configure_smoke_logging()
    attach_run_log_file(extra_logger_names=("__main__",))


def main(argv: list[str] | None = None) -> int:
    load_dotenv(find_dotenv())
    args = parse_args(argv)

    if args.gate != "m7.2":
        print(f"Unsupported gate {args.gate!r}; only m7.2 is implemented", file=sys.stderr)
        return 2

    run_id = _utc_run_id()
    run_dir = REPO_ROOT / "artifacts" / "colab" / MILESTONE / run_id
    run_log = run_dir / "run.log"
    _configure_logging(run_log)

    branch, sha = git_metadata()
    entrypoint = str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    argv_list = sys.argv if argv is None else [entrypoint] + list(argv)

    seed = args.seed
    if seed is None:
        raw_seed = os.environ.get("ROBOCREW_SIM_SEED", "").strip()
        seed = int(raw_seed) if raw_seed else 0

    status = "fail"
    reason = "not started"
    metrics: dict[str, Any] = {}
    drift_m: float | None = None

    try:
        logger.info(
            "M7.2 probe start: suite=%s task_idx=%s steps=%s seed=%s",
            args.task_suite,
            args.task_idx,
            args.steps,
            seed,
        )
        primary = run_wiring_probe(
            task_suite=args.task_suite,
            task_idx=args.task_idx,
            steps=args.steps,
            task_prompt=args.task_prompt,
            server_url=args.vla_url,
            seed=seed,
        )
        metrics["primary"] = primary

        if args.drift_check:
            repeat_steps = max(3, min(args.steps, 5))
            repeat = run_wiring_probe(
                task_suite=args.task_suite,
                task_idx=args.task_idx,
                steps=repeat_steps,
                task_prompt=args.task_prompt,
                server_url=args.vla_url,
                seed=seed,
            )
            metrics["repeat"] = repeat
            drift_m = _drift_between_runs(primary, repeat)
            metrics["drift_m"] = drift_m

        status, reason = evaluate_m7_2(
            completed_without_error=primary.get("completed_without_error", False),
            displacement=primary.get("displacement_m"),
            min_displacement=args.min_displacement,
        )
    except Exception:
        reason = traceback.format_exc(limit=12)
        logger.error("M7.2 probe failed:\n%s", reason)

    summary = {
        "run_id": run_id,
        "milestone": MILESTONE,
        "gate": args.gate,
        "timestamp_utc": run_id,
        "git_branch": branch,
        "git_sha": sha,
        "entrypoint": entrypoint,
        "argv": argv_list,
        "env_snapshot": collect_env_snapshot(),
        "status": status,
        "reason": reason,
        "metrics": {
            **metrics,
            "min_displacement_m": args.min_displacement,
            "task_suite": args.task_suite,
            "task_idx": args.task_idx,
            "steps_requested": args.steps,
        },
        "artifacts": {
            "run_dir": str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
            "run_log": str(run_log.relative_to(REPO_ROOT)).replace("\\", "/"),
            "summary_json": str((run_dir / "summary.json").relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
        },
    }
    summary_path = write_summary(run_dir, summary)
    logger.info("Wrote %s status=%s reason=%s", summary_path, status, reason)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
