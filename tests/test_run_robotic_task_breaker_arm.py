"""Hermetic tests for validation/run_robotic_task_breaker_arm.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARM_PATH = _REPO_ROOT / "validation" / "run_robotic_task_breaker_arm.py"


def _load_arm():
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location(
        "run_robotic_task_breaker_arm", _ARM_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_episode_writes_summary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arm = _load_arm()
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

    monkeypatch.setattr(arm, "Runner", _FakeRunner)
    monkeypatch.setattr(arm, "configure_logging", lambda: None)

    sys.argv = [
        "run_robotic_task_breaker_arm.py", "--episodes", "2",
        "--artifact-dir", str(tmp_path)]
    assert arm.main() == 0
    records_path = tmp_path / "robotic_task_breaker_episodes.jsonl"
    records = [
        json.loads(line) for line in records_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 2
    assert all(row["arm"] == "robotic_task_breaker" for row in records)
    assert all(row["task_description"] == task_desc for row in records)
    assert all(row["success"] is True for row in records)
    assert all(row["stop_reason"] == arm._RUNNER_STOP_PLACEHOLDER
               for row in records)
    summary_path = tmp_path / "robotic_task_breaker_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["episodes"] == 2
    assert summary["successes"] == 2


def test_run_episode_runner_init_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arm = _load_arm()

    class _BrokenRunner:
        def __init__(self, _task: str, _task_idx: int = 0) -> None:
            raise RuntimeError("sim unavailable")

    monkeypatch.setattr(arm, "Runner", _BrokenRunner)
    monkeypatch.setattr(arm, "configure_logging", lambda: None)

    row = arm._run_episode(0)
    assert row["success"] is False
    assert row["stop_reason"] == "execution_error"
    assert "sim unavailable" in str(row["error"])
