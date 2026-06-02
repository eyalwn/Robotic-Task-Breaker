from __future__ import annotations

from core.plan import ExecutionResult, Executor, FeedbackDecision, Plan
from core.skills import Skill, SkillRegistry


class MockSkill(Skill):
    name = "mock.skill"
    description = "mock skill"
    parameters = []
    returns = "Any"

    def __init__(self, return_value: object = None) -> None:
        self.return_value = return_value

    def __call__(self, **kwargs: object) -> object:
        return self.return_value


class MockPlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self._plans = plans
        self.calls: list[tuple[str, str, str]] = []

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        self.calls.append((goal, catalog, context))
        if self._plans:
            return self._plans.pop(0)
        raise RuntimeError("No plan available")


class SequenceFeedbackEvaluator:
    def __init__(self, decisions: list[FeedbackDecision]) -> None:
        self._decisions = decisions
        self._idx = 0

    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackDecision:
        if self._idx >= len(self._decisions):
            return self._decisions[-1]
        decision = self._decisions[self._idx]
        self._idx += 1
        return decision


def mock_image_fn() -> bytes:
    return b"fake-image"


def make_executor(
    plans: list[Plan],
    decisions: list[FeedbackDecision],
    step_budget: int = 30,
    max_replans: int = 3,
) -> tuple[Executor, SkillRegistry, MockPlanner]:
    registry = SkillRegistry()
    registry.register(MockSkill("ok"))
    planner = MockPlanner(plans)
    executor = Executor(
        registry=registry,
        planner=planner,
        feedback_evaluator=SequenceFeedbackEvaluator(decisions),
        image_fn=mock_image_fn,
        max_replans=max_replans,
        step_budget=step_budget,
    )
    return executor, registry, planner


def test_executor_done() -> None:
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "call", "skill": "mock.skill"}})
    executor, _, _ = make_executor([], [FeedbackDecision.CONTINUE])
    result = executor.run("g", plan)
    assert isinstance(result, ExecutionResult)
    assert result.status == "done"
    assert result.value is None


def test_executor_return_value() -> None:
    plan = Plan.model_validate(
        {"goal": "g", "body": {"kind": "return", "value": {"kind": "literal", "value": 99}}}
    )
    executor, _, _ = make_executor([], [FeedbackDecision.CONTINUE])
    result = executor.run("g", plan)
    assert isinstance(result, ExecutionResult)
    assert result.status == "done"
    assert result.value == 99


def test_executor_stopped() -> None:
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "stop now"}})
    executor, _, _ = make_executor([], [FeedbackDecision.STOP])
    result = executor.run("g", plan)
    assert result.status == "stopped"
    assert result.value is None


def test_executor_replan_then_done() -> None:
    first = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry"}})
    second = Plan.model_validate({"goal": "g", "body": {"kind": "call", "skill": "mock.skill"}})
    executor, _, planner = make_executor([second], [FeedbackDecision.REPLAN, FeedbackDecision.CONTINUE])
    result = executor.run("g", first)
    assert result.status == "done"
    assert result.replans == 1
    assert len(planner.calls) == 1


def test_executor_replan_limit() -> None:
    first = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry"}})
    second = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry-again"}})
    executor, _, _ = make_executor([second], [FeedbackDecision.REPLAN, FeedbackDecision.REPLAN], max_replans=1)
    result = executor.run("g", first)
    assert result.status == "replan_limit"


def test_executor_step_budget() -> None:
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill"},
                    {"kind": "call", "skill": "mock.skill"},
                    {"kind": "call", "skill": "mock.skill"},
                ],
            },
        }
    )
    executor, _, _ = make_executor([], [FeedbackDecision.CONTINUE], step_budget=2)
    result = executor.run("g", plan)
    assert result.status == "step_budget"
