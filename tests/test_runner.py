"""Runner orchestration tests with mocked planner and executor."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from executor import ExecutionSum, Executor, StopReason
from plan import Plan
from runner import Runner


def _done_plan() -> Plan:
    return []


def _stop_plan(reason: str) -> dict[str, str]:
    return {"stop": reason}


class _StubPlanner:
    def __init__(self, goal: str = "", catalog: str = "") -> None:
        self.init_goal = goal
        self.init_catalog = catalog
        self.plan_calls = 0
        self.plan_args: list[bytes] = []
        self._plan_fail_on_call = -1
        self._stop_on_call = -1
        self._stop_reason = "goal achieved"

    def fail_on_plan_call(self, call_number: int) -> None:
        self._plan_fail_on_call = call_number

    def stop_on_plan_call(self, call_number: int, reason: str = "goal achieved") -> None:
        self._stop_on_call = call_number
        self._stop_reason = reason

    def plan(self, snapshot: bytes) -> object:
        self.plan_calls += 1
        self.plan_args.append(snapshot)
        if self._plan_fail_on_call == self.plan_calls:
            raise RuntimeError("plan validation failed")
        if self._stop_on_call == self.plan_calls:
            return _stop_plan(self._stop_reason)
        return _done_plan()


class _StubEnv:
    task_description = (
        "put both the alphabet soup and the tomato sauce in the basket"
    )

    def __init__(self, *, success: bool = True) -> None:
        self._success = success
        self._steps = 0
        self.closed = False
        self.events: list[str] = []
        self.observe_calls = 0

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self) -> tuple[tuple[bytes, bytes, bytes], list[float]]:
        self.observe_calls += 1
        return (b"img", b"second", b"wrist"), [0.0] * 7

    def step(self, action: list[float]) -> bool:
        self._steps += 1
        return True

    def success(self) -> bool:
        self.events.append("success")
        return self._success

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


def _runner_with_stub_executor(
    summaries: list[ExecutionSum],
    monkeypatch: pytest.MonkeyPatch,
    *,
    planner=None,
    env_success: bool = True,
    runner_args: tuple = ("select_fruit",),
) -> tuple[Runner, _StubPlanner, MagicMock]:
    planner = planner or _StubPlanner()
    env = _StubEnv(success=env_success)
    executor = MagicMock(spec=Executor)
    summary_iter = iter(summaries)

    def _run(_steps: Plan) -> ExecutionSum:
        return next(summary_iter)

    executor.run.side_effect = _run

    def _planner_factory(goal: str, catalog: str, **_kwargs: object) -> _StubPlanner:
        planner.init_goal = goal
        planner.init_catalog = catalog
        return planner

    monkeypatch.setattr("runner.SimEnv", lambda *args, **kwargs: env)
    monkeypatch.setattr("runner.LLMPlanner", _planner_factory)
    monkeypatch.setattr("runner.Executor", lambda *_args, **_kwargs: executor)

    runner = Runner(*runner_args)
    return runner, planner, executor


def test_runner_uses_env_task_description_for_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, planner, _ = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE)],
        monkeypatch,
    )
    assert runner.run() is True
    assert planner.init_goal == _StubEnv.task_description


def test_runner_args_reach_sim_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture_sim(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _StubEnv(success=True)

    monkeypatch.setattr("runner.SimEnv", _capture_sim)
    monkeypatch.setattr(
        "runner.LLMPlanner",
        lambda goal, catalog, **_kwargs: _StubPlanner(goal, catalog),
    )
    monkeypatch.setattr(
        "runner.Executor",
        lambda *_args, **_kwargs: MagicMock(
            run=lambda _steps: ExecutionSum(reason=StopReason.PLAN_DONE)
        ),
    )
    Runner("select_fruit", 0).run()
    assert captured["args"] == ("select_fruit", 0)


def test_runner_plan_done_success(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE)],
        monkeypatch,
    )
    assert runner.run() is True
    assert planner.plan_calls == 1
    assert planner.plan_args[0] == b"img"
    assert executor.run.call_count == 1


def test_runner_run_calls_success_before_close(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE)],
        monkeypatch,
        env_success=True,
    )
    assert runner.run() is True
    env = runner._env
    assert "success" in env.events
    assert env.events.index("success") < env.events.index("close")


def test_runner_planner_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    planner.stop_on_plan_call(1, reason="goal achieved")
    runner, _, executor = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE)],
        monkeypatch,
        planner=planner,
    )
    assert runner.run() is False
    assert executor.run.call_count == 0


def test_runner_subtask_timeout_from_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.SUBTASK_TIMEOUT)],
        monkeypatch,
    )
    assert runner.run() is False
