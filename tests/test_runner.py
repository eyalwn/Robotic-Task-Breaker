"""Runner orchestration tests with mocked planner and executor."""
from __future__ import annotations

from typing import Any

from unittest.mock import MagicMock

import pytest

from executor import ExecutionStop, Executor, StopReason
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
    def __init__(
        self,
        goal: str = "",
        catalog: str = "",
        scene_objects: list | None = None,
    ) -> None:
        self.init_goal = goal
        self.init_scene_objects = scene_objects
        self.plan_calls = 0
        self.plan_args: list[tuple[ExecutionStop | None, bytes | None]] = []
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
        stop: ExecutionStop | None = None,
        snapshot: bytes | None = None,
    ) -> dict[str, Any]:
        self.plan_calls += 1
        self.plan_args.append((stop, snapshot))
        if self._plan_fail_on_call == self.plan_calls:
            raise RuntimeError("plan validation failed")
        if self._stop_on_call == self.plan_calls:
            return _stop_plan_dict(self._stop_reason)
        return _done_plan_dict(self.init_goal)


class _StubEnv:
    def __init__(self, *, success: bool = True) -> None:
        self._success = success
        self.closed = False
        self.events: list[str] = []
        self.observe_calls = 0

    def observe(self) -> tuple[bytes, bytes, list[float]]:
        self.observe_calls += 1
        return b"img", b"wrist", [0.0] * 8

    def step(self, action: list[float]) -> bool:
        return True

    def success(self) -> bool:
        self.events.append("success")
        return self._success

    def object_names(self) -> list[str]:
        return ["test_object_1"]

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


def _runner_with_stub_executor(
    stops: list[ExecutionStop],
    monkeypatch: pytest.MonkeyPatch,
    *,
    planner: _StubPlanner | None = None,
    env_success: bool = True,
) -> tuple[Runner, _StubPlanner, MagicMock]:
    planner = planner or _StubPlanner()
    env = _StubEnv(success=env_success)
    executor = MagicMock(spec=Executor)
    stop_iter = iter(stops)

    def _run(_plan: Plan) -> ExecutionStop:
        return next(stop_iter)

    executor.run.side_effect = _run

    def _planner_factory(goal: str, _catalog: str, scene_objects: list, **_kwargs: object) -> _StubPlanner:
        planner.init_goal = goal
        planner.init_scene_objects = scene_objects
        return planner

    monkeypatch.setattr("runner.LiberoEnv", lambda *_args, **_kwargs: env)
    monkeypatch.setattr("runner.LLMPlanner", _planner_factory)
    monkeypatch.setattr("runner.Executor", lambda *_args, **_kwargs: executor)

    runner = Runner("test goal", "libero_10", 0)
    return runner, planner, executor


def test_runner_passes_raw_goal_to_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, _ = _runner_with_stub_executor(
        [ExecutionStop(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
    )
    assert runner.run() is None
    assert planner.init_goal == "test goal"


def test_runner_plan_done_success(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [ExecutionStop(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
    )
    status, error, value, total_steps = runner._run_loop()
    assert runner._verdict(status, value) is True
    assert status == "done"
    assert error == ""
    assert planner.plan_calls == 1
    assert planner.plan_args[0] == (None, None)
    assert executor.run.call_count == 1
    assert total_steps == 0


def test_runner_run_calls_success_before_close(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [ExecutionStop(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
        env_success=True,
    )
    assert runner.run() is None
    env = runner._env
    assert "success" in env.events
    assert env.events.index("success") < env.events.index("close")


def test_runner_feedback_triggers_replan_then_done(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    runner, planner, executor = _runner_with_stub_executor(
        [
            ExecutionStop(
                reason=StopReason.FEEDBACK_POINT,
                steps=2,
                feedback_description="checkpoint",
                feedback_prompt="checkpoint",
            ),
            ExecutionStop(reason=StopReason.PLAN_DONE, return_value="done"),
        ],
        monkeypatch,
        planner=planner,
    )
    status, error, value, _ = runner._run_loop()
    assert runner._verdict(status, value) is True
    assert planner.plan_calls == 2
    stop, snapshot = planner.plan_args[1]
    assert stop is not None
    assert stop.feedback_description == "checkpoint"
    assert snapshot == b"img"
    assert runner._env.observe_calls == 1
    assert executor.run.call_count == 2
    assert error == ""


def test_runner_multiple_feedback_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    feedback = ExecutionStop(
        reason=StopReason.FEEDBACK_POINT,
        feedback_description="checkpoint",
        feedback_prompt="checkpoint",
    )
    runner, planner, executor = _runner_with_stub_executor(
        [feedback, feedback, ExecutionStop(reason=StopReason.PLAN_DONE, return_value="done")],
        monkeypatch,
        planner=planner,
    )
    status, error, value, _ = runner._run_loop()
    assert runner._verdict(status, value) is True
    assert planner.plan_calls == 3
    assert planner.plan_args[0] == (None, None)
    assert planner.plan_args[1][0] is not None
    assert planner.plan_args[1][1] == b"img"
    assert planner.plan_args[2][0] is not None
    assert planner.plan_args[2][1] == b"img"
    assert runner._env.observe_calls == 2
    assert executor.run.call_count == 3
    assert error == ""


def test_runner_feedback_plan_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    planner.fail_on_plan_call(2)
    runner, _, _ = _runner_with_stub_executor(
        [
            ExecutionStop(
                reason=StopReason.FEEDBACK_POINT,
                feedback_description="checkpoint",
                feedback_prompt="checkpoint",
            ),
        ],
        monkeypatch,
        planner=planner,
    )
    assert runner.run() is None


def test_runner_planner_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    planner.stop_on_plan_call(2, reason="goal achieved")
    runner, _, _ = _runner_with_stub_executor(
        [
            ExecutionStop(
                reason=StopReason.FEEDBACK_POINT,
                feedback_description="checkpoint",
                feedback_prompt="checkpoint",
            ),
        ],
        monkeypatch,
        planner=planner,
    )
    status, error, value, _ = runner._run_loop()
    assert runner._verdict(status, value) is False
    assert status == "stopped"
    assert error == "goal achieved"
    assert value == ""


def test_runner_step_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [ExecutionStop(reason=StopReason.STEP_BUDGET)],
        monkeypatch,
    )
    status, error, value, _ = runner._run_loop()
    assert status == "step_budget"
    assert runner._verdict(status, value) is False
    assert error == ""


def test_runner_feedback_missing_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [ExecutionStop(reason=StopReason.FEEDBACK_POINT)],
        monkeypatch,
    )
    status, error, value, _ = runner._run_loop()
    assert status == "execution_error"
    assert error == "feedback stop missing description or prompt"
    assert value == ""
