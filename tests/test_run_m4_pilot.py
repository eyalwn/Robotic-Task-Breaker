"""Hermetic tests for validation/run_m4_pilot.py Longor episode records."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from executor import ExecutionSum, StopReason

_REPO_ROOT = Path(__file__).resolve().parent.parent
_M4_PATH = _REPO_ROOT / "validation" / "run_m4_pilot.py"


def _load_m4_module():
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location("run_m4_pilot", _M4_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_longor_status_fields_clear_on_success() -> None:
    m4 = _load_m4_module()
    stop, err = m4._final_longor_status_fields(
        True, StopReason.SUBTASK_TIMEOUT, "skill timed out"
    )
    assert stop == ""
    assert err == ""


def test_final_longor_status_fields_keep_failure() -> None:
    m4 = _load_m4_module()
    stop, err = m4._final_longor_status_fields(
        False, StopReason.SUBTASK_TIMEOUT, "skill timed out"
    )
    assert stop == StopReason.SUBTASK_TIMEOUT
    assert err == "skill timed out"


def test_longor_episode_retry_timeout_then_success(monkeypatch) -> None:
    m4 = _load_m4_module()
    plan = [
        {"skill": "open", "object_name": "top drawer"},
        {"skill": "close", "object_name": "top drawer"},
    ]
    timeout_sum = ExecutionSum(
        reason=StopReason.SUBTASK_TIMEOUT,
        steps=400,
        error="close timed out",
        next_step_idx=1,
    )
    done_sum = ExecutionSum(
        reason=StopReason.PLAN_DONE,
        steps=460,
        error="",
    )
    executor = MagicMock()
    executor.run.side_effect = [timeout_sum, done_sum]

    class FakeEnv:
        task_description = "open the top drawer, then close it"
        steps = 0

        def observe(self):
            return (["img"], [])

        def success(self):
            return True

        def close(self):
            return None

    planner = MagicMock()
    planner.plan.return_value = plan

    monkeypatch.setattr(m4, "SimEnv", lambda *_a, **_k: FakeEnv())
    monkeypatch.setattr(m4, "LLMPlanner", lambda *_a, **_k: planner)
    monkeypatch.setattr(m4, "Executor", lambda *_a, **_k: executor)
    monkeypatch.setattr(m4.config, "MAX_REPLANS", 1)

    row = m4._run_longor_episode()

    assert row["success"] is True
    assert row["stop_reason"] == ""
    assert row["error"] == ""
    assert row["steps"] == 860
    assert planner.plan.call_count == 1
    assert executor.run.call_count == 2
    assert len(row["attempts"]) == 2
    assert row["attempts"][0]["stop_reason"] == StopReason.SUBTASK_TIMEOUT
    assert row["attempts"][0]["error"] == "close timed out"
    assert row["attempts"][0]["next_step_idx"] == 1
    assert [step["skill"] for step in row["attempts"][0]["plan"]] == [
        "open",
        "close",
    ]
    assert row["attempts"][1]["stop_reason"] == StopReason.PLAN_DONE
    assert row["attempts"][1]["steps"] == 460
    assert [step["skill"] for step in row["attempts"][1]["plan"]] == ["close"]
