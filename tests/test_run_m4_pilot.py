"""Hermetic tests for validation/run_m4_pilot.py paired pilot orchestration."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from executor import ExecutionSum, StopReason
from envs.sim_env import SimEnv

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


def test_longor_episode_builds_planner_before_executor(monkeypatch) -> None:
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

    def _planner(*_a, **_k):
        order.append("planner")
        return MagicMock(plan=lambda *_p: {"stop": "n/a"})

    exec_mock = MagicMock()

    def _executor(*_a, **_k):
        order.append("executor")
        return exec_mock

    monkeypatch.setattr(m4, "LLMPlanner", _planner)
    monkeypatch.setattr(m4, "Executor", _executor)
    m4._run_longor_episode(FakeEnv())
    assert order == ["planner", "executor"]
    exec_mock.run.assert_not_called()


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

    row = m4._run_longor_episode(FakeEnv())

    assert row["success"] is True
    assert row["stop_reason"] == ""
    assert row["error"] == ""
    assert row["steps"] == 860
    assert planner.plan.call_count == 1
    assert executor.run.call_count == 1
    assert len(row["attempts"]) == 1
    assert row["attempts"][0]["stop_reason"] == StopReason.PLAN_DONE
    assert row["attempts"][0]["skill_telemetry"][1]["threshold_crossed"] is True


def test_arm_order_counterbalanced() -> None:
    m4 = _load_m4_module()
    assert m4._arm_order_for_pair(0) == (m4.ARM_BARE, m4.ARM_LONGOR)
    assert m4._arm_order_for_pair(1) == (m4.ARM_LONGOR, m4.ARM_BARE)
    assert m4._arm_order_for_pair(2) == (m4.ARM_BARE, m4.ARM_LONGOR)


def test_parse_arms_requires_both() -> None:
    m4 = _load_m4_module()
    assert m4._parse_arms("bare,longor") == ["bare", "longor"]
    with pytest.raises(RuntimeError, match="requires exactly both"):
        m4._parse_arms("bare")


def test_observability_mismatch_reason() -> None:
    m4 = _load_m4_module()
    first = {"task_description": "a", "state_vector": "b"}
    second = {"task_description": "x", "state_vector": "b"}
    reason = m4._observability_mismatch_reason(first, second)
    assert "task_description" in reason
    assert m4._observability_mismatch_reason(first, first) == ""


class _PairedFakeBackend:
    restore_count = 0
    mismatch_on_restore = False

    def __init__(self) -> None:
        self._steps = 0
        self.task_description = "open the top drawer, then close it"
        self._drawer_was_opened = False

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def horizon(self) -> int:
        return 1000

    def observe(self):
        tag = b"arm-a" if not _PairedFakeBackend.mismatch_on_restore else b"arm-b"
        return ((tag, tag, tag), [1.0, 2.0])

    def step(self, _action):
        self._steps += 1
        return False

    def success(self):
        return True

    def check_subtask(self, _step):
        return False

    def drawer_joint_state(self):
        return {"door": 0.0}

    def episode_wrapper_state(self):
        return {"steps": self._steps, "drawer_was_opened": self._drawer_was_opened}

    def capture_episode_checkpoint(self):
        return {"version": 1, "tag": "ckpt"}

    def restore_episode_checkpoint(self, _checkpoint):
        _PairedFakeBackend.restore_count += 1

    def close(self):
        return None


def test_sim_env_forwards_checkpoint_methods() -> None:
    backend = _PairedFakeBackend()
    env = SimEnv.__new__(SimEnv)
    env._env = backend
    assert env.capture_episode_checkpoint() == {"version": 1, "tag": "ckpt"}
    env.restore_episode_checkpoint({"version": 1})
    assert _PairedFakeBackend.restore_count == 1


def test_paired_pilot_disables_video_until_arm_run(
    monkeypatch, tmp_path) -> None:
    m4 = _load_m4_module()
    save_video_at_simenv: list[bool] = []
    save_video_at_run: list[bool] = []

    def _sim(*_a, **_k):
        save_video_at_simenv.append(m4.config.SAVE_VIDEO)
        return _PairedFakeBackend()

    def _bare(_env):
        save_video_at_run.append(m4.config.SAVE_VIDEO)
        return {"success": True, "category": m4.CATEGORY_SUCCESS,
                "task_description": "open the top drawer, then close it",
                "target_object": "top drawer", "prompt": "x",
                "stop_reason": "", "steps": 1, "error": ""}

    def _longor(_env):
        save_video_at_run.append(m4.config.SAVE_VIDEO)
        return {"success": False, "category": m4.CATEGORY_PLANNER_PLAN,
                "task_description": "open the top drawer, then close it",
                "target_object": "top drawer", "plan_status": "stopped",
                "stop_reason": "n/a", "steps": 0, "error": "",
                "last_plan": None, "attempts": []}

    monkeypatch.setattr(m4.config, "SAVE_VIDEO", True)
    monkeypatch.setattr(m4, "SimEnv", _sim)
    monkeypatch.setattr(m4, "_run_bare_episode", _bare)
    monkeypatch.setattr(m4, "_run_longor_episode", _longor)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_m4_pilot.py", "--episodes", "1",
         "--artifact-dir", str(tmp_path), "--arms", "bare,longor"],
    )
    assert m4.main() == 0
    assert save_video_at_simenv == [False, False, False]
    assert save_video_at_run == [True, True]
    assert m4.config.SAVE_VIDEO is True


def test_paired_pilot_writes_provenance_and_exits_on_mismatch(
    monkeypatch, tmp_path) -> None:
    m4 = _load_m4_module()
    _PairedFakeBackend.restore_count = 0
    _PairedFakeBackend.mismatch_on_restore = False

    def _sim(*_a, **_k):
        return _PairedFakeBackend()

    def _bare(env):
        return {"success": True, "category": m4.CATEGORY_SUCCESS,
                "task_description": env.task_description,
                "target_object": "top drawer", "prompt": env.task_description,
                "stop_reason": "episode_timeout", "steps": 1, "error": ""}

    def _longor(env):
        return {"success": False, "category": m4.CATEGORY_PLANNER_PLAN,
                "task_description": env.task_description,
                "target_object": "top drawer", "plan_status": "stopped",
                "stop_reason": "n/a", "steps": 0, "error": "",
                "last_plan": None, "attempts": []}

    monkeypatch.setattr(m4, "SimEnv", _sim)
    monkeypatch.setattr(m4, "_run_bare_episode", _bare)
    monkeypatch.setattr(m4, "_run_longor_episode", _longor)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_m4_pilot.py", "--episodes", "1",
         "--artifact-dir", str(tmp_path), "--arms", "bare,longor"],
    )
    assert m4.main() == 0
    rows = [json.loads(line) for line in
            (tmp_path / "m4_pilot_episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {row["pair_id"] for row in rows} == {0}
    assert rows[0]["arm_order"] == "bare,longor"
    assert rows[0]["pair_valid"] is True
    assert "initial_hashes" in rows[0]
    assert rows[0]["initial_hashes"] == rows[1]["initial_hashes"]


def test_paired_pilot_fail_closed_on_observability_mismatch(
    monkeypatch, tmp_path) -> None:
    m4 = _load_m4_module()
    call_idx = {"n": 0}

    class _DriftingBackend(_PairedFakeBackend):
        def observe(self):
            call_idx["n"] += 1
            tag = b"same" if call_idx["n"] <= 1 else b"diff"
            return ((tag, tag, tag), [1.0, 2.0])

    def _sim(*_a, **_k):
        return _DriftingBackend()

    def _bare(env):
        return {"success": True, "category": m4.CATEGORY_SUCCESS,
                "task_description": env.task_description,
                "target_object": "top drawer", "prompt": env.task_description,
                "stop_reason": "", "steps": 1, "error": ""}

    def _longor(_env):
        raise AssertionError("longor must not run after observability mismatch")

    monkeypatch.setattr(m4, "SimEnv", _sim)
    monkeypatch.setattr(m4, "_run_bare_episode", _bare)
    monkeypatch.setattr(m4, "_run_longor_episode", _longor)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_m4_pilot.py", "--episodes", "1",
         "--artifact-dir", str(tmp_path), "--arms", "bare,longor"],
    )
    assert m4.main() == 1
    rows = [json.loads(line) for line in
            (tmp_path / "m4_pilot_episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["pair_valid"] is False for row in rows)
    assert rows[1]["stop_reason"] == "pair_invalid"
