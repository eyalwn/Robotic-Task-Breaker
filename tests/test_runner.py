"""Runner orchestration tests with mocked planner and executor."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from executor import BoundaryReason, ExecutionBoundary, Executor
from plan import Plan
from planner import FeedbackHandoff, PlannerAction, PlannerDecision
from runner import Runner
from skills import SkillRegistry


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
        self.decide_calls = 0
        self._decisions: list[PlannerDecision] = []

    def queue_decision(self, decision: PlannerDecision) -> None:
        self._decisions.append(decision)

    def plan(self, context: str = "") -> Plan:
        self.plan_calls += 1
        self.last_context = context
        return Plan.model_validate(
            {
                "goal": self.init_goal,
                "body": {"kind": "return", "value": {"kind": "literal", "value": "done"}},
            }
        )

    def decide(self, handoff: FeedbackHandoff) -> PlannerDecision:
        self.decide_calls += 1
        if not self._decisions:
            return PlannerDecision(action=PlannerAction.CONTINUE)
        return self._decisions.pop(0)


class _StubEnv:
    def __init__(self, *, success: bool = True) -> None:
        self._success = success
        self.closed = False

    def observe(self) -> tuple[bytes, bytes, list[float]]:
        return b"img", b"wrist", [0.0] * 8

    def step(self, action: list[float]) -> bool:
        return True

    def success(self) -> bool:
        return self._success

    def object_names(self) -> list[str]:
        return ["test_object_1"]

    def close(self) -> None:
        self.closed = True


def _runner_with_stub_executor(
    boundaries: list[ExecutionBoundary],
    monkeypatch: pytest.MonkeyPatch,
    *,
    planner: _StubPlanner | None = None,
    env_success: bool = True,
) -> tuple[Runner, _StubPlanner, MagicMock]:
    planner = planner or _StubPlanner()
    env = _StubEnv(success=env_success)
    executor = MagicMock(spec=Executor)
    boundary_iter = iter(boundaries)

    def _run() -> ExecutionBoundary:
        return next(boundary_iter)

    executor.run.side_effect = _run
    executor.reset = MagicMock()

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
        [ExecutionBoundary(reason=BoundaryReason.PLAN_DONE, return_value="done")],
        monkeypatch,
    )
    runner.run()
    assert planner.init_goal == "test goal"


def test_runner_plan_done_success(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, planner, executor = _runner_with_stub_executor(
        [ExecutionBoundary(reason=BoundaryReason.PLAN_DONE, return_value="done")],
        monkeypatch,
    )
    result = runner.run()
    assert result.success is True
    assert result.status == "done"
    assert planner.plan_calls == 1
    assert executor.reset.call_count == 1
    assert executor.run.call_count == 1


def test_runner_feedback_continue_then_done(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    planner.queue_decision(PlannerDecision(action=PlannerAction.CONTINUE))
    runner, planner, executor = _runner_with_stub_executor(
        [
            ExecutionBoundary(
                reason=BoundaryReason.FEEDBACK_POINT,
                feedback_description="checkpoint",
                feedback_prompt="checkpoint",
                image_bytes=b"img",
            ),
            ExecutionBoundary(reason=BoundaryReason.PLAN_DONE, return_value="done"),
        ],
        monkeypatch,
        planner=planner,
    )
    result = runner.run()
    assert result.success is True
    assert planner.decide_calls == 1
    assert executor.run.call_count == 2


def test_runner_feedback_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    planner.queue_decision(PlannerDecision(action=PlannerAction.STOP, reason="halt"))
    runner, _, _ = _runner_with_stub_executor(
        [
            ExecutionBoundary(
                reason=BoundaryReason.FEEDBACK_POINT,
                feedback_description="halt",
                feedback_prompt="halt",
                image_bytes=b"img",
            ),
        ],
        monkeypatch,
        planner=planner,
    )
    result = runner.run()
    assert result.success is False
    assert result.status == "stopped"
    assert result.error == "halt"


def test_runner_replan_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _StubPlanner()
    new_plan = Plan.model_validate(
        {
            "goal": "retry",
            "body": {"kind": "return", "value": {"kind": "literal", "value": "x"}},
        }
    )
    for _ in range(4):
        planner.queue_decision(
            PlannerDecision(action=PlannerAction.NEW_PLAN, plan=new_plan)
        )

    feedback = ExecutionBoundary(
        reason=BoundaryReason.FEEDBACK_POINT,
        feedback_description="retry",
        feedback_prompt="retry",
        image_bytes=b"img",
    )
    runner, _, executor = _runner_with_stub_executor(
        [feedback, feedback, feedback, feedback],
        monkeypatch,
        planner=planner,
    )
    result = runner.run()
    assert result.status == "replan_limit"
    assert result.replans == 3
    assert executor.reset.call_count == 4


def test_runner_step_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _, _ = _runner_with_stub_executor(
        [ExecutionBoundary(reason=BoundaryReason.STEP_BUDGET)],
        monkeypatch,
    )
    result = runner.run()
    assert result.status == "step_budget"
    assert result.success is False
