"""Runner orchestration tests with mocked planner and executor."""
from __future__ import annotations

from typing import Any

from unittest.mock import MagicMock

import pytest

from executor import ExecutionSum, Executor, StopReason
from plan import Plan
from runner import Runner


def _done_plan_dict(goal: str) -> dict[str, Any]:
    return {
        "goal": goal,
        "body": {
            "kind": "return",
            "value": {"kind": "literal", "value": "done"},
        },
    }


def _stop_plan_dict(reason: str) -> dict[str, Any]:
    return {
        "goal": "stop",
        "body": {
            "kind": "return",
            "value": {"kind": "literal", "value": f"STOP: {reason}"},
        },
    }


class _StubPlanner:
    def __init__(self, goal: str = "", catalog: str = "") -> None:
        self.init_goal = goal
        self.init_catalog = catalog
        self.plan_calls = 0
        self.plan_args: list[tuple[bytes, ExecutionSum | None]] = []
        self._plan_fail_on_call: int | None = None
        self._stop_on_call: int | None = None
        self._stop_reason = "goal achieved"

    def fail_on_plan_call(self, call_number: int) -> None:
        self._plan_fail_on_call = call_number

    def stop_on_plan_call(self, call_number: int, reason: str = "goal achieved") -> None:
        self._stop_on_call = call_number
        self._stop_reason = reason

    def plan(
        self,
        snapshot: bytes,
        execution_sum: ExecutionSum | None = None,
    ) -> dict[str, Any]:
        self.plan_calls += 1
        self.plan_args.append((snapshot, execution_sum))
        if self._plan_fail_on_call == self.plan_calls:
            raise RuntimeError("plan validation failed")
        if self._stop_on_call == self.plan_calls:
            return _stop_plan_dict(self._stop_reason)
        return _done_plan_dict(self.init_goal)


class _StubEnv:
    def __init__(self, *, success: bool = True, max_steps: int = 520) -> None:
        self._success = success
        self._steps = 0
        self.max_steps = max_steps
        self.closed = False
        self.events: list[str] = []
        self.observe_calls = 0

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self) -> tuple[bytes, bytes, list[float]]:
        self.observe_calls += 1
        return b"img", b"wrist", [0.0] * 8

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
    planner: _StubPlanner | None = None,
    env_success: bool = True,
    max_steps: int = 520,
) -> tuple[Runner, _StubPlanner, MagicMock]:
    planner = planner or _StubPlanner()
    env = _StubEnv(success=env_success, max_steps=max_steps)
    executor = MagicMock(spec=Executor)
    summary_iter = iter(summaries)

    def _run(_plan: Plan) -> ExecutionSum:
        return next(summary_iter)

    executor.run.side_effect = _run

    def _planner_factory(goal: str, catalog: str, **_kwargs: object) -> _StubPlanner:
        planner.init_goal = goal
        planner.init_catalog = catalog
        return planner

    monkeypatch.setattr("runner.LiberoEnv", lambda *_args, **_kwargs: env)
    monkeypatch.setattr("runner.LLMPlanner", _planner_factory)
    monkeypatch.setattr("runner.Executor", lambda *_args, **_kwargs: executor)

    runner = Runner("test goal", "libero_10", 0)
    return runner, planner, executor


def test_runner_passes_raw_goal_to_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, _ = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
    )
    assert runner.run() is None
    assert planner.init_goal == "test goal"


def test_runner_plan_done_success(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
    )
    status, error, value, total_steps = runner._execute_once()
    assert runner._verdict(status, value) is True
    assert status == "done"
    assert error == ""
    assert planner.plan_calls == 1
    assert planner.plan_args[0] == (b"img", None)
    assert executor.run.call_count == 1
    assert total_steps == 0


def test_runner_run_calls_success_before_close(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
        env_success=True,
    )
    assert runner.run() is None
    env = runner._env
    assert "success" in env.events
    assert env.events.index("success") < env.events.index("close")


def test_runner_feedback_checkpoint_is_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [
            ExecutionSum(
                reason=StopReason.FEEDBACK_POINT,
                steps=2,
                feedback_description="checkpoint",
                feedback_prompt="checkpoint",
            ),
        ],
        monkeypatch,
    )
    status, error, value, _ = runner._execute_once()
    assert runner._verdict(status, value) is False
    assert status == "execution_error"
    assert error == "plan hit feedback checkpoint (unsupported in single-pass mode)"
    assert planner.plan_calls == 1
    assert executor.run.call_count == 1
    assert runner._env.observe_calls == 1


def test_runner_planner_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    planner.stop_on_plan_call(1, reason="goal achieved")
    runner, _, executor = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
        planner=planner,
    )
    status, error, value, _ = runner._execute_once()
    assert runner._verdict(status, value) is False
    assert status == "stopped"
    assert error == "goal achieved"
    assert value == ""
    assert executor.run.call_count == 0


def test_runner_step_budget_from_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.STEP_BUDGET)],
        monkeypatch,
    )
    status, error, value, _ = runner._execute_once()
    assert status == "step_budget"
    assert error == "executor step budget exceeded"
    assert runner._verdict(status, value) is False


def test_runner_episode_steps_exhausted_before_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [ExecutionSum(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
        max_steps=0,
    )
    status, error, value, _ = runner._execute_once()
    assert status == "step_budget"
    assert error == "episode step budget exhausted before plan"
    assert planner.plan_calls == 0
    assert executor.run.call_count == 0
