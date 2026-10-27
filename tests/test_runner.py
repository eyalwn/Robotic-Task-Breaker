"""Runner orchestration tests with mocked planner and executor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import config
from executor import ExecutionSum, Executor, StopReason
from plan import Plan
from runner import Runner


def _sum(reason: str, error: str = "") -> ExecutionSum:
    return ExecutionSum(
        reason=reason, steps=0, error=error, next_step_idx=0, skill_telemetry=[])


def _done_plan() -> Plan:
    return [{"skill": "open", "object_name": "right drawer"},
            {"skill": "close", "object_name": "right drawer"}]


def _stop_plan(reason: str) -> dict[str, str]:
    return {"stop": reason}


class _StubPlanner:
    def __init__(self, catalog: str = "") -> None:
        self.init_catalog = catalog
        self.plan_calls = 0
        self.plan_goals: list[str] = []
        self.plan_args: list[bytes] = []
        self._plan_output: object = _done_plan()
        self._plan_fail_on_call = -1
        self._stop_on_call = -1
        self._stop_reason = "goal achieved"

    def set_plan_output(self, output: object) -> None:
        self._plan_output = output

    def fail_on_plan_call(self, call_number: int) -> None:
        self._plan_fail_on_call = call_number

    def stop_on_plan_call(
            self, call_number: int, reason: str = "goal achieved") -> None:
        self._stop_on_call = call_number
        self._stop_reason = reason

    def plan(self, goal: str, snapshot: bytes) -> object:
        self.plan_calls += 1
        self.plan_goals.append(goal)
        self.plan_args.append(snapshot)
        if self._plan_fail_on_call == self.plan_calls:
            raise RuntimeError("plan validation failed")
        if self._stop_on_call == self.plan_calls:
            return _stop_plan(self._stop_reason)
        return self._plan_output


class _StubEnv:
    task_description = (
        "put both the alphabet soup and the tomato sauce in the basket")

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
        summaries: list[ExecutionSum], monkeypatch: pytest.MonkeyPatch, *,
        planner=None, env_success: bool = True,
        runner_args: tuple = ("select_fruit",)) -> tuple[
                Runner, _StubPlanner, MagicMock]:
    planner = planner or _StubPlanner()
    env = _StubEnv(success=env_success)
    executor = MagicMock(spec=Executor)
    summary_iter = iter(summaries)

    def _run(_steps: Plan) -> ExecutionSum:
        return next(summary_iter)

    executor.run.side_effect = _run

    def _planner_factory(catalog: str, **_kwargs: object) -> _StubPlanner:
        planner.init_catalog = catalog
        return planner

    monkeypatch.setattr("runner.SimEnv", lambda *args, **kwargs: env)
    monkeypatch.setattr("runner.LLMPlanner", _planner_factory)
    monkeypatch.setattr("runner.Executor", lambda *_args, **_kwargs: executor)

    runner = Runner(*runner_args)
    return runner, planner, executor


def test_runner_uses_env_task_description_for_planner(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, _ = _runner_with_stub_executor(
        [_sum(StopReason.PLAN_DONE)], monkeypatch)
    assert runner.run() is True
    assert planner.plan_goals == [_StubEnv.task_description]


def test_runner_initializes_planner_before_sim_env(
        monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    def _sim(*_args, **_kwargs) -> _StubEnv:
        order.append("sim")
        return _StubEnv(success=True)

    def _planner(_catalog: str, **_kwargs: object) -> _StubPlanner:
        order.append("planner")
        return _StubPlanner()

    done = _sum(StopReason.PLAN_DONE)
    monkeypatch.setattr("runner.SimEnv", _sim)
    monkeypatch.setattr("runner.LLMPlanner", _planner)
    monkeypatch.setattr(
        "runner.Executor", lambda *_a, **_k: MagicMock(run=lambda _s, **_kw: done))
    Runner("select_fruit", 0).run()
    assert order == ["planner", "sim"]


def test_runner_args_reach_sim_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture_sim(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _StubEnv(success=True)

    monkeypatch.setattr("runner.SimEnv", _capture_sim)
    monkeypatch.setattr(
        "runner.LLMPlanner", lambda catalog, **_kwargs: _StubPlanner(catalog))
    done = _sum(StopReason.PLAN_DONE)
    monkeypatch.setattr("runner.Executor",
                        lambda *_a, **_k: MagicMock(run=lambda _steps, **_kw: done))
    Runner("select_fruit", 0).run()
    assert captured["args"] == ("select_fruit", 0)


def test_runner_plan_done_success(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [_sum(StopReason.PLAN_DONE)], monkeypatch)
    assert runner.run() is True
    assert planner.plan_calls == 1
    assert planner.plan_args[0] == b"img"
    assert executor.run.call_count == 1


def test_runner_run_calls_success_before_close(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [_sum(StopReason.PLAN_DONE)], monkeypatch, env_success=True)
    assert runner.run() is True
    env = runner._env
    assert "success" in env.events
    assert env.events.index("success") < env.events.index("close")


def test_runner_planner_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    planner.stop_on_plan_call(1, reason="goal achieved")
    runner, _, executor = _runner_with_stub_executor(
        [_sum(StopReason.PLAN_DONE)], monkeypatch,
        planner=planner, env_success=False)
    assert runner.run() is False
    assert executor.run.call_count == 0


def test_runner_subtask_timeout_exhausts_replans(
        monkeypatch: pytest.MonkeyPatch) -> None:
    n = config.MAX_REPLANS + 1
    runner, planner, executor = _runner_with_stub_executor(
        [_sum(StopReason.SUBTASK_TIMEOUT)] * n, monkeypatch, env_success=False)
    assert runner.run() is False
    assert planner.plan_calls == n
    assert executor.run.call_count == n


def test_runner_replans_after_subtask_timeout(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [_sum(StopReason.SUBTASK_TIMEOUT, "open exceeded 2 steps"),
         _sum(StopReason.PLAN_DONE)], monkeypatch)
    assert runner.run() is True
    assert planner.plan_calls == 2
    assert executor.run.call_count == 2
    assert executor.run.call_args.kwargs == {}


def test_runner_episode_timeout_does_not_replan(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [_sum(StopReason.EPISODE_TIMEOUT,
              "episode hits maximum steps-budget: 2 steps")], monkeypatch,
        env_success=False)
    assert runner.run() is False
    assert planner.plan_calls == 1
    assert executor.run.call_count == 1


def test_runner_episode_timeout_uses_terminal_success_verdict(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [_sum(StopReason.EPISODE_TIMEOUT,
              "episode hits maximum steps-budget: 2 steps")], monkeypatch,
        env_success=True)
    assert runner.run() is True
    assert planner.plan_calls == 1
    assert executor.run.call_count == 1


def test_runner_execution_error_does_not_replan(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [_sum(StopReason.EXECUTION_ERROR, "boom")], monkeypatch, env_success=False)
    assert runner.run() is False
    assert planner.plan_calls == 1
    assert executor.run.call_count == 1
