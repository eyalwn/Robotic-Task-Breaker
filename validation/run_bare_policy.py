"""Run bare-VLA episodes on ``open_close_drawer`` (M4 bare arm).

Each episode prompts the policy with the raw task language and steps until
horizon or env done. Writes JSONL episode records and a compact summary JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from envs.robocasa_env import _OPEN_CLOSE_SUFFIX  # noqa: E402
from envs.sim_env import SimEnv  # noqa: E402
from utils import configure_logging  # noqa: E402
from vla_client import VLAClient  # noqa: E402

_TASK = "open_close_drawer"
_RECORDS_NAME = "bare_episodes.jsonl"
_SUMMARY_NAME = "bare_summary.json"


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


def _object_name_from_task_description(task_desc: str) -> str:
    """Parse drawer object name from ``open_close_drawer`` task description."""
    if not task_desc.endswith(_OPEN_CLOSE_SUFFIX):
        raise RuntimeError(
            f"task_description must end with {_OPEN_CLOSE_SUFFIX!r}, "
            f"got {task_desc!r}")
    open_phrase = task_desc[: -len(_OPEN_CLOSE_SUFFIX)].strip()
    prefix = "open the "
    if not open_phrase.lower().startswith(prefix):
        raise RuntimeError(
            f"expected reset language 'open the <name>', got {open_phrase!r}")
    name = open_phrase[len(prefix) :].strip()
    if not name:
        raise RuntimeError(
            f"empty object name in task description {task_desc!r}")
    return name


def _run_episode(episode_index: int) -> dict[str, object]:
    """Run one bare-policy episode and return a JSON-serializable record."""
    env = SimEnv(_TASK, 0)
    task_desc = env.task_description
    error = ""
    stop_reason = "episode_timeout"
    try:
        object_name = _object_name_from_task_description(task_desc)
        open_step = {"skill": "open", "object_name": object_name}
        prompt = task_desc
        client = VLAClient()
        done = False
        while env.steps < env.horizon and not done:
            images, state = env.observe()
            for action in client.predict(images, state, prompt):
                done = env.step(action)
                if done or env.steps >= env.horizon:
                    break
        success = env.success()
        if not success and not env.check_subtask(open_step):
            stop_reason = "open_not_achieved"
        elif done:
            stop_reason = "episode_done"
        else:
            stop_reason = "episode_timeout"
        return {
            "arm": "bare",
            "episode_index": episode_index,
            "task_description": task_desc,
            "success": success,
            "steps": env.steps,
            "stop_reason": stop_reason,
            "error": error,
        }
    except Exception as exc:
        error = str(exc)
        return {
            "arm": "bare",
            "episode_index": episode_index,
            "task_description": task_desc,
            "success": False,
            "steps": env.steps,
            "stop_reason": "execution_error",
            "error": error,
        }
    finally:
        env.close()


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
        "arm": "bare",
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
        print(f"bare episode={ep_idx} success={row['success']} "
              f"steps={row['steps']} stop_reason={row['stop_reason']!r} "
              f"error={row['error']!r}")
    summary = _write_artifacts(args.artifact_dir, new_rows, args.append)
    print(f"summary episodes={summary['episodes']} "
          f"successes={summary['successes']} "
          f"rate={summary['success_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
