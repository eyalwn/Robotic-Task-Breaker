"""Run Longor ``Runner`` episodes on ``open_close_drawer`` (M4 Longor arm).

Writes JSONL episode records and a compact summary JSON under one artifact dir.
Each episode uses ``Runner.run()`` (observe, plan, execute, and replan
after a subtask timeout).

``Runner`` returns only the final success verdict; episode records use
``steps=0`` and ``stop_reason='not_reported_by_runner'`` as explicit
placeholders (see ``_RUNNER_STOP_PLACEHOLDER``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from runner import Runner  # noqa: E402
from utils import configure_logging  # noqa: E402

_TASK = "open_close_drawer"
_RECORDS_NAME = "longor_episodes.jsonl"
_SUMMARY_NAME = "longor_summary.json"
_RUNNER_STOP_PLACEHOLDER = "not_reported_by_runner"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append episodes to existing JSONL and refresh summary.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First episode index for record metadata (default 0).",
    )
    return parser.parse_args()


def _run_episode(episode_index: int) -> dict[str, object]:
    """Run one Longor episode via ``Runner`` and return a JSON record."""
    task_desc = ""
    try:
        runner = Runner(_TASK, 0)
        task_desc = runner._goal
        success = runner.run()
        return {
            "arm": "longor",
            "episode_index": episode_index,
            "task_description": task_desc,
            "success": success,
            "steps": 0,
            "stop_reason": _RUNNER_STOP_PLACEHOLDER,
            "error": "",
        }
    except Exception as exc:
        return {
            "arm": "longor",
            "episode_index": episode_index,
            "task_description": task_desc,
            "success": False,
            "steps": 0,
            "stop_reason": "execution_error",
            "error": str(exc),
        }


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    successes = sum(1 for row in rows if bool(row.get("success")))
    total = len(rows)
    rate = (successes / total) if total else 0.0
    return {
        "arm": "longor",
        "task": _TASK,
        "episodes": total,
        "successes": successes,
        "success_rate": rate,
    }


def _write_artifacts(
        artifact_dir: Path,
        new_rows: list[dict[str, object]],
        append: bool) -> dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / _RECORDS_NAME
    summary_path = artifact_dir / _SUMMARY_NAME
    rows = _load_records(records_path) if append else []
    rows.extend(new_rows)
    with records_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    summary = _summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    configure_logging()
    args = _parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    new_rows: list[dict[str, object]] = []
    for offset in range(args.episodes):
        ep_idx = args.start_index + offset
        row = _run_episode(ep_idx)
        new_rows.append(row)
        print(f"longor episode={ep_idx} success={row['success']} "
              f"steps={row['steps']} stop_reason={row['stop_reason']!r} "
              f"error={row['error']!r}")
    summary = _write_artifacts(args.artifact_dir, new_rows, args.append)
    print(f"summary episodes={summary['episodes']} "
          f"successes={summary['successes']} "
          f"rate={summary['success_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
