"""Write M4 parity-gate evaluation artifacts (baseline vs Longor subtask).

Used by Colab notebooks; not part of the runtime orchestration path.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VLABENCH_SHA = "cf588fe60c0c7282174fe979f5913170cfe69017"
DEFAULT_POLICY = "lerobot/smolvla_vlabench"
DEFAULT_TASK = "select_fruit"


def _git_output(args: list[str], cwd: Path) -> str:
    """Return stripped stdout from a git command, or empty string on failure."""
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _package_version(name: str) -> str:
    """Return an installed package version string when available."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    """Return a CUDA device label when nvidia-smi is available."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out.splitlines()[0] if out else "none"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"


def build_metadata(
    *,
    track: str,
    task: str = DEFAULT_TASK,
    policy: str = DEFAULT_POLICY,
    n_episodes: int,
    successes: int,
    repo_root: Path,
    log_path: Path,
    videos_dir: Path | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble one parity-gate artifact payload."""
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    if not 0 <= successes <= n_episodes:
        raise ValueError(
            f"successes must be in [0, {n_episodes}], got {successes}"
        )
    payload = {
        "gate": "m4_parity",
        "track": track,
        "task": task,
        "policy": policy,
        "vlabench_sha": VLABENCH_SHA,
        "lerobot_version": _package_version("lerobot"),
        "longor_commit": _git_output(["rev-parse", "HEAD"], repo_root),
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "gpu": _gpu_name(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
    }
    if videos_dir is not None:
        payload["videos_dir"] = str(videos_dir)
    if extra:
        payload.update(extra)
    return payload


def write_artifact(path: Path, payload: dict) -> Path:
    """Persist one artifact JSON file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def apply_stop_condition(baseline: dict, longor: dict) -> str:
    """Return the parity-gate stop-condition label for two artifact payloads."""
    baseline_rate = float(baseline["success_rate"])
    longor_rate = float(longor["success_rate"])
    if baseline_rate <= 0.0:
        return "official_reference_fails"
    if longor_rate > 0.0:
        return "repaired_longor_succeeds"
    return "longor_fails_baseline_succeeds"


def main() -> None:
    """CLI: write artifact from args or print stop condition from two JSON files."""
    import argparse

    parser = argparse.ArgumentParser(description="M4 parity artifact helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    write_cmd = sub.add_parser("write")
    write_cmd.add_argument("--track", required=True)
    write_cmd.add_argument("--task", default=DEFAULT_TASK)
    write_cmd.add_argument("--policy", default=DEFAULT_POLICY)
    write_cmd.add_argument("--n-episodes", type=int, required=True)
    write_cmd.add_argument("--successes", type=int, required=True)
    write_cmd.add_argument("--out", type=Path, required=True)
    write_cmd.add_argument("--log", type=Path, required=True)
    write_cmd.add_argument("--videos-dir", type=Path, default=None)
    write_cmd.add_argument("--repo-root", type=Path, default=Path("."))

    verdict_cmd = sub.add_parser("verdict")
    verdict_cmd.add_argument("--baseline", type=Path, required=True)
    verdict_cmd.add_argument("--longor", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "write":
        payload = build_metadata(
            track=args.track,
            task=args.task,
            policy=args.policy,
            n_episodes=args.n_episodes,
            successes=args.successes,
            repo_root=args.repo_root.resolve(),
            log_path=args.log.resolve(),
            videos_dir=args.videos_dir.resolve() if args.videos_dir else None,
        )
        write_artifact(args.out.resolve(), payload)
        print(json.dumps(payload, indent=2))
        return

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    longor = json.loads(args.longor.read_text(encoding="utf-8"))
    label = apply_stop_condition(baseline, longor)
    print(label)


if __name__ == "__main__":
    main()
