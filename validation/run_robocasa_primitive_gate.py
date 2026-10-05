"""Run the upstream RoboCasa primitive-policy suitability gate (M1).

This script owns orchestration and verdict logic for the upstream-only baseline
gate. It intentionally does not import Longor runtime modules.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


READINESS_LOG_MARKER = "Server is ready and listening on"
DEFAULT_ENV_NAME = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
DEFAULT_MODEL_PATH = "nvidia/GR00T-N1.6-3B"
DEFAULT_EMBODIMENT_TAG = "ROBOCASA_PANDA_OMRON"
DEFAULT_POLICY_PORT = 5555


@dataclass
class GateResult:
    """Structured verdict result for one primitive-gate run."""

    passed: bool
    successes: int
    total: int
    success_rate: float
    threshold_successes: int
    note: str
    rollout_exit_code: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run upstream GR00T primitive gate and write verdict artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Absolute artifact directory path (will be created if missing).",
    )
    parser.add_argument(
        "--groot-root",
        default="/content/Isaac-GR00T",
        help="Path to Isaac-GR00T checkout with built environments.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embodiment-tag", default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--policy-port", type=int, default=DEFAULT_POLICY_PORT)
    parser.add_argument("--readiness-timeout-s", type=int, default=900)
    parser.add_argument("--readiness-poll-s", type=int, default=5)
    parser.add_argument("--readiness-min-port-polls", type=int, default=3)
    parser.add_argument(
        "--threshold-successes",
        type=int,
        default=7,
        help="Pass threshold over the rollout episodes.",
    )
    return parser.parse_args()


def _port_accepts_loopback(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _start_server(
    *,
    groot_root: Path,
    model_path: str,
    embodiment_tag: str,
    server_log_path: Path,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = (
        f'cd "{groot_root}" && '
        "source .venv/bin/activate && "
        "uv run python gr00t/eval/run_gr00t_server.py "
        f"--model-path {model_path} "
        f"--embodiment-tag {embodiment_tag} "
        "--use-sim-policy-wrapper"
    )
    log_fp = server_log_path.open("ab")
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc


def _wait_for_readiness(
    *,
    proc: subprocess.Popen[bytes],
    policy_port: int,
    server_log_path: Path,
    timeout_s: int,
    poll_s: int,
    min_port_polls: int,
) -> tuple[bool, str, int]:
    deadline = time.monotonic() + timeout_s
    port_ready_streak = 0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            reason = (
                "policy server exited before readiness "
                f"(exit={proc.returncode})")
            return False, reason, port_ready_streak

        log_text = server_log_path.read_text(encoding="utf-8", errors="replace")
        log_ready = READINESS_LOG_MARKER in log_text
        port_ready = _port_accepts_loopback(policy_port)
        port_ready_streak = port_ready_streak + 1 if port_ready else 0

        if log_ready and port_ready:
            reason = (
                f"log marker {READINESS_LOG_MARKER!r} observed and "
                f"127.0.0.1:{policy_port} accepts connections"
            )
            return True, reason, port_ready_streak

        if port_ready_streak >= min_port_polls:
            reason = (
                f"port 127.0.0.1:{policy_port} accepted connections for "
                f"{port_ready_streak} consecutive polls (log marker may be buffered)"
            )
            return True, reason, port_ready_streak

        time.sleep(poll_s)

    marker_seen = READINESS_LOG_MARKER in server_log_path.read_text(
        encoding="utf-8", errors="replace")
    return (
        False,
        (
            f"readiness timeout after {timeout_s}s: "
            "log_marker_seen="
            f"{marker_seen}; "
            f"port_open={_port_accepts_loopback(policy_port)}; "
            f"port_ready_streak={port_ready_streak}"
        ),
        port_ready_streak,
    )


def _run_rollout(
    *,
    groot_root: Path,
    episodes: int,
    policy_port: int,
    max_episode_steps: int,
    env_name: str,
    n_action_steps: int,
    n_envs: int,
    rollout_log_path: Path,
) -> int:
    py_bin = groot_root / "gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
    cmd = [
        str(py_bin),
        "gr00t/eval/rollout_policy.py",
        "--n_episodes",
        str(episodes),
        "--policy_client_host",
        "127.0.0.1",
        "--policy_client_port",
        str(policy_port),
        "--max_episode_steps",
        str(max_episode_steps),
        "--env_name",
        env_name,
        "--n_action_steps",
        str(n_action_steps),
        "--n_envs",
        str(n_envs),
    ]
    with rollout_log_path.open("wb") as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=groot_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_fp.write(chunk)
            log_fp.flush()
        proc.wait()
        return int(proc.returncode)


def _parse_rollout_result(
    log_text: str, expected_episodes: int) -> tuple[int, int, str]:
    episode_hits = re.findall(
        r"episode\s*\d+.*?\bsuccess\b\s*[:=]\s*(true|false|1|0)",
        log_text,
        flags=re.IGNORECASE,
    )
    if episode_hits:
        successes = sum(token.lower() in ("true", "1") for token in episode_hits)
        return successes, len(episode_hits), "per-episode success flags"

    summary_hits = re.findall(
        r"success(?:es)?\s*[:=]\s*(\d+)\s*/\s*(\d+)",
        log_text,
        flags=re.IGNORECASE,
    )
    if summary_hits:
        success_str, total_str = summary_hits[-1]
        return int(success_str), int(total_str), "summary successes/total pattern"

    rate_hits = re.findall(
        r"success[_\s-]*rate\s*[:=]\s*([0-9]*\.?[0-9]+)",
        log_text,
        flags=re.IGNORECASE,
    )
    if rate_hits:
        rate = float(rate_hits[-1])
        successes = int(round(rate * expected_episodes))
        return successes, expected_episodes, "summary success_rate pattern"

    return (
        0,
        expected_episodes,
        "no parseable success summary; treated as 0 successes")


def _build_verdict(
    *,
    rollout_exit_code: int,
    rollout_log_path: Path,
    threshold_successes: int,
    expected_episodes: int,
) -> GateResult:
    log_text = rollout_log_path.read_text(encoding="utf-8", errors="replace")
    successes, total, note = _parse_rollout_result(log_text, expected_episodes)
    success_rate = float(successes) / float(total) if total > 0 else 0.0
    passed = rollout_exit_code == 0 and successes >= threshold_successes
    return GateResult(
        passed=passed,
        successes=successes,
        total=total,
        success_rate=success_rate,
        threshold_successes=threshold_successes,
        note=note,
        rollout_exit_code=rollout_exit_code,
    )


def _collect_artifacts(
    artifact_dir: Path, rollout_log_path: Path, server_log_path: Path) -> None:
    upstream_outputs = artifact_dir / "upstream_outputs"
    upstream_outputs.mkdir(parents=True, exist_ok=True)

    for src in (rollout_log_path, server_log_path):
        if src.is_file():
            shutil.copy2(src, upstream_outputs / src.name)

    log_text = rollout_log_path.read_text(encoding="utf-8", errors="replace")
    video_match = re.search(r"Video saved to:\s*(\S+)", log_text)
    if video_match:
        video_src = Path(video_match.group(1))
        if video_src.is_dir():
            shutil.copytree(
                video_src, upstream_outputs / "videos", dirs_exist_ok=True)


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    subprocess.run(["pkill", "-f", "run_gr00t_server.py"], check=False)
    try:
        os.kill(proc.pid, signal.SIGTERM)
    except Exception:
        pass


def main() -> int:
    args = _parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    groot_root = Path(args.groot_root)

    server_log_path = artifact_dir / "policy_server.log"
    rollout_log_path = artifact_dir / "rollout.log"
    readiness_path = artifact_dir / "server_readiness.json"
    rollout_code_path = artifact_dir / "rollout_exit_code.txt"
    verdict_json_path = artifact_dir / "verdict.json"
    verdict_txt_path = artifact_dir / "verdict.txt"

    server_log_path.write_text("", encoding="utf-8")
    rollout_log_path.write_text("", encoding="utf-8")

    started = time.monotonic()
    proc = _start_server(
        groot_root=groot_root,
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        server_log_path=server_log_path,
    )
    print(f"policy_server_pid={proc.pid}")
    try:
        ready, reason, final_streak = _wait_for_readiness(
            proc=proc,
            policy_port=args.policy_port,
            server_log_path=server_log_path,
            timeout_s=args.readiness_timeout_s,
            poll_s=args.readiness_poll_s,
            min_port_polls=args.readiness_min_port_polls,
        )
        readiness = {
            "ready": ready,
            "reason": reason,
            "timeout_s": args.readiness_timeout_s,
            "port": args.policy_port,
            "log_marker": READINESS_LOG_MARKER,
            "min_port_ready_polls": args.readiness_min_port_polls,
            "final_port_ready_streak": final_streak,
            "elapsed_s": round(time.monotonic() - started, 2),
            "server_pid": proc.pid,
        }
        readiness_path.write_text(
            json.dumps(readiness, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(readiness, indent=2))

        if not ready:
            rollout_code_path.write_text("-1\n", encoding="utf-8")
            verdict = GateResult(
                passed=False,
                successes=0,
                total=args.episodes,
                success_rate=0.0,
                threshold_successes=args.threshold_successes,
                note=f"server not ready: {reason}",
                rollout_exit_code=-1,
            )
        else:
            rollout_exit_code = _run_rollout(
                groot_root=groot_root,
                episodes=args.episodes,
                policy_port=args.policy_port,
                max_episode_steps=args.max_episode_steps,
                env_name=args.env_name,
                n_action_steps=args.n_action_steps,
                n_envs=args.n_envs,
                rollout_log_path=rollout_log_path,
            )
            rollout_code_path.write_text(f"{rollout_exit_code}\n", encoding="utf-8")
            verdict = _build_verdict(
                rollout_exit_code=rollout_exit_code,
                rollout_log_path=rollout_log_path,
                threshold_successes=args.threshold_successes,
                expected_episodes=args.episodes,
            )

        verdict_json = {
            "passed": verdict.passed,
            "successes": verdict.successes,
            "total": verdict.total,
            "success_rate": verdict.success_rate,
            "threshold_successes": verdict.threshold_successes,
            "note": verdict.note,
            "rollout_exit_code": verdict.rollout_exit_code,
            "artifact_dir": str(artifact_dir),
        }
        verdict_json_path.write_text(
            json.dumps(verdict_json, indent=2) + "\n", encoding="utf-8")
        verdict_txt = (
            f"{'PASS' if verdict.passed else 'FAIL'} "
            f"successes={verdict.successes}/{verdict.total} "
            f"success_rate={verdict.success_rate:.3f} "
            f"threshold_successes={verdict.threshold_successes} "
            f"rollout_exit_code={verdict.rollout_exit_code} "
            f"note={verdict.note}"
        )
        verdict_txt_path.write_text(verdict_txt + "\n", encoding="utf-8")
        print(verdict_txt)

        _collect_artifacts(artifact_dir, rollout_log_path, server_log_path)
        return 0 if verdict.passed else 1
    finally:
        _stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(main())
