"""Hermetic tests for validation/run_longor_arm.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LONGOR_ARM_PATH = _REPO_ROOT / "validation" / "run_longor_arm.py"


def _load_longor_arm():
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location(
        "run_longor_arm", _LONGOR_ARM_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_episode_writes_summary(tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    longor_arm = _load_longor_arm()
    task_desc = "open the right drawer, then close it"

    class _FakeRunner:
        def __init__(self, task: str, task_idx: int = 0) -> None:
            assert task == "open_close_drawer"
            assert task_idx == 0
            self._goal = task_desc
            self.runs = 0

        def run(self) -> bool:
            self.runs += 1
            return True

    monkeypatch.setattr(longor_arm, "Runner", _FakeRunner)
    monkeypatch.setattr(longor_arm, "configure_logging", lambda: None)

    sys.argv = [
        "run_longor_arm.py",
        "--episodes",
        "2",
        "--artifact-dir",
        str(tmp_path),
    ]
    assert longor_arm.main() == 0
    records = [
        json.loads(line)
        for line in (tmp_path / "longor_episodes.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert all(row["arm"] == "longor" for row in records)
    assert all(row["task_description"] == task_desc for row in records)
    assert all(row["success"] is True for row in records)
    assert all(
        row["stop_reason"] == longor_arm._RUNNER_STOP_PLACEHOLDER
        for row in records)
    summary = json.loads(
        (tmp_path / "longor_summary.json").read_text(encoding="utf-8"))
    assert summary["episodes"] == 2
    assert summary["successes"] == 2


def test_run_episode_runner_init_error(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    longor_arm = _load_longor_arm()

    class _BrokenRunner:
        def __init__(self, _task: str, _task_idx: int = 0) -> None:
            raise RuntimeError("sim unavailable")

    monkeypatch.setattr(longor_arm, "Runner", _BrokenRunner)
    monkeypatch.setattr(longor_arm, "configure_logging", lambda: None)

    row = longor_arm._run_episode(0)
    assert row["success"] is False
    assert row["stop_reason"] == "execution_error"
    assert "sim unavailable" in str(row["error"])
