"""Hermetic tests for validation/run_m4_pilot.py Longor episode records."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

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
        True, StopReason.EPISODE_TIMEOUT, "episode timed out"
    )
    assert stop == ""
    assert err == ""


def test_final_longor_status_fields_keep_failure() -> None:
    m4 = _load_m4_module()
    stop, err = m4._final_longor_status_fields(
        False, StopReason.EPISODE_TIMEOUT, "episode timed out"
    )
    assert stop == StopReason.EPISODE_TIMEOUT
    assert err == "episode timed out"


def test_longor_episode_builds_planner_before_sim_env(monkeypatch) -> None:
    m4 = _load_m4_module()
    order: list[str] = []

    class FakeEnv:
        task_description = "open the top drawer, then close it"
        steps = 0

        def observe(self):
            return (["img"], [])

        def success(self):
            return False

        def close(self):
            return None

    def _sim(*_a, **_k):
        order.append("sim")
        return FakeEnv()

    def _planner(*_a, **_k):
        order.append("planner")
        return MagicMock(plan=lambda *_p: {"stop": "n/a"})

    monkeypatch.setattr(m4, "SimEnv", _sim)
    monkeypatch.setattr(m4, "LLMPlanner", _planner)
    m4._run_longor_episode()
    assert order == ["planner", "sim"]


def test_longor_episode_plan_once_execute_once(monkeypatch) -> None:
    m4 = _load_m4_module()
    plan = [
        {"skill": "open", "object_name": "top drawer"},
        {"skill": "close", "object_name": "top drawer"},
    ]
    done_sum = ExecutionSum(
        reason=StopReason.PLAN_DONE,
        steps=860,
        error="",
        skill_telemetry=[
            {"skill": "open", "steps": 400, "threshold_crossed": False},
            {"skill": "close", "steps": 460, "threshold_crossed": True},
        ],
    )
    executor = MagicMock()
    executor.run.return_value = done_sum

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

    row = m4._run_longor_episode()

    assert row["success"] is True
    assert row["stop_reason"] == ""
    assert row["error"] == ""
    assert row["steps"] == 860
    assert planner.plan.call_count == 1
    assert executor.run.call_count == 1
    assert len(row["attempts"]) == 1
    assert row["attempts"][0]["stop_reason"] == StopReason.PLAN_DONE
    assert row["attempts"][0]["skill_telemetry"][1]["threshold_crossed"] is True
