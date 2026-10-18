"""Hermetic tests for validation/run_drawer_chain_gate.py diagnostics."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from executor import ExecutionSum, StopReason

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GATE_PATH = _REPO_ROOT / "validation" / "run_drawer_chain_gate.py"


def _load_gate_module():
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location("run_drawer_chain_gate", _GATE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEnv:
    task_description = "open the right drawer, then close it"

    def __init__(self) -> None:
        self._steps = 0
        self._horizon = 1000
        self.closed = False

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def horizon(self) -> int:
        return self._horizon

    def check_subtask(self, step: dict[str, str]) -> bool:
        return step["skill"] in ("open", "close")

    def drawer_joint_state(self) -> dict[str, float]:
        return {"right_joint": 0.02}

    def success(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def _build_executor_factory(gate_module, env: _FakeEnv, seen_budgets: list[int]):
    class _FakeExecutor:
        def __init__(self, env) -> None:
            self._env = env

        def _build_vla_task(self, step: dict[str, str]) -> str:
            return f"{step['skill']} the {step['object_name']}"

        def run(self, steps):
            skill = steps[0]["skill"]
            if skill == "open":
                self._env._steps = 588
                return ExecutionSum(reason=StopReason.PLAN_DONE, steps=188)
            seen_budgets.append(gate_module.config.MAX_SKILL_STEPS["close"])
            return ExecutionSum(reason=StopReason.PLAN_DONE, steps=200)

    return _FakeExecutor


def test_run_episode_default_close_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _load_gate_module()
    env = _FakeEnv()
    seen_budgets: list[int] = []
    executor_cls = _build_executor_factory(gate, env, seen_budgets)
    monkeypatch.setattr(gate, "SimEnv", lambda *_a, **_k: env)
    monkeypatch.setattr(gate, "Executor", executor_cls)

    original_budgets = dict(gate.config.MAX_SKILL_STEPS)
    row = gate._run_episode(close_to_horizon=False)

    assert row["close_budget"] == original_budgets["close"]
    assert seen_budgets == [original_budgets["close"]]
    assert gate.config.MAX_SKILL_STEPS == original_budgets
    assert row["final_joints"]["right_joint"] == pytest.approx(0.02)
    assert env.closed is True


def test_run_episode_close_to_horizon_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _load_gate_module()
    env = _FakeEnv()
    seen_budgets: list[int] = []
    executor_cls = _build_executor_factory(gate, env, seen_budgets)
    monkeypatch.setattr(gate, "SimEnv", lambda *_a, **_k: env)
    monkeypatch.setattr(gate, "Executor", executor_cls)

    original_budgets = dict(gate.config.MAX_SKILL_STEPS)
    row = gate._run_episode(close_to_horizon=True)

    assert row["close_budget"] == 412
    assert seen_budgets == [412]
    assert gate.config.MAX_SKILL_STEPS == original_budgets
    assert row["close_stop_reason"] == StopReason.PLAN_DONE
