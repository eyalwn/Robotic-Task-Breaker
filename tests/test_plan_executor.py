from __future__ import annotations

from core.plan import ExecutionResult, Executor, FeedbackDecision, FeedbackOutcome, Plan, validate_plan
from core.skills import Skill, SkillRegistry


class ArmGrabMockSkill(Skill):
    name = "arm.grab"
    description = "mock arm grab"
    parameters = []
    returns = "str"

    def __init__(self, returns: list[str]) -> None:
        self._returns = returns
        self._idx = 0

    def __call__(self, **kwargs: object) -> str:
        if self._idx >= len(self._returns):
            return self._returns[-1]
        value = self._returns[self._idx]
        self._idx += 1
        return value


class MockSkill(Skill):
    name = "mock.skill"
    description = "mock skill"
    parameters = []
    returns = "Any"

    def __init__(self, return_value: object = None) -> None:
        self.return_value = return_value

    def __call__(self, **kwargs: object) -> object:
        return self.return_value


class StaticPlanMockPlanner:
    def __init__(self, plan_to_return: Plan) -> None:
        self._plan_to_return = plan_to_return

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        return self._plan_to_return


class MockPlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self._plans = plans
        self.calls: list[tuple[str, str, str]] = []

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        self.calls.append((goal, catalog, context))
        if self._plans:
            return self._plans.pop(0)
        raise RuntimeError("No plan available")


class FailingPlanner:
    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        raise RuntimeError("LLMPlanner: plan validation failed after 3 attempts")


class SequenceFeedbackEvaluator:
    def __init__(self, decisions: list[FeedbackDecision]) -> None:
        self._decisions = decisions
        self._idx = 0

    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackOutcome:
        if self._idx >= len(self._decisions):
            return FeedbackOutcome(decision=self._decisions[-1])
        decision = self._decisions[self._idx]
        self._idx += 1
        return FeedbackOutcome(decision=decision)


def mock_image_fn() -> bytes:
    return b"fake-image"


def handcrafted_plan() -> Plan:
    return validate_plan(
        {
            "goal": "pick up the cube",
            "body": {
                "kind": "seq",
                "steps": [
                    {
                        "kind": "call",
                        "skill": "arm.grab",
                        "args": {"object_name": {"kind": "literal", "value": "cube"}},
                        "assign_to": "grab_result",
                    },
                    {"kind": "feedback", "description": "Is the cube grasped?"},
                    {"kind": "return", "value": {"kind": "var", "name": "grab_result"}},
                ],
            },
        }
    )


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


def test_executor_replan_planner_failure() -> None:
    first = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry"}})
    registry = SkillRegistry()
    registry.register(MockSkill("ok"))
    executor = Executor(
        registry=registry,
        planner=FailingPlanner(),
        feedback_evaluator=SequenceFeedbackEvaluator([FeedbackDecision.REPLAN]),
        image_fn=mock_image_fn,
        max_replans=3,
        step_budget=30,
    )
    result = executor.run("g", first)
    assert result.status == "replan_failed"
    assert result.replans == 1
    assert "plan validation failed" in (result.error or "")


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


def test_collect_plan_skills() -> None:
    from core.plan.executor import _collect_plan_skills

    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "perception.detect"},
                    {"kind": "call", "skill": "arm.grab"},
                    {"kind": "feedback", "description": "grasped?"},
                    {"kind": "call", "skill": "arm.place"},
                ],
            },
        }
    )
    assert _collect_plan_skills(plan.body) == [
        "perception.detect",
        "arm.grab",
        "arm.place",
    ]


def test_plan_start_trace_includes_skills(tmp_path) -> None:
    import json

    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "feedback", "description": "halt"},
                    {"kind": "call", "skill": "perception.detect"},
                    {"kind": "call", "skill": "arm.grab"},
                    {"kind": "call", "skill": "arm.place"},
                ],
            },
        }
    )
    registry = SkillRegistry()
    registry.register(MockSkill("ok"))
    executor = Executor(
        registry=registry,
        planner=MockPlanner([]),
        feedback_evaluator=SequenceFeedbackEvaluator([FeedbackDecision.STOP]),
        image_fn=mock_image_fn,
        trace_dir=str(tmp_path),
    )
    executor.run("g", plan)

    trace_files = list(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 1
    first_event = json.loads(trace_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert first_event["event"] == "plan_start"
    assert first_event["skills"] == [
        "perception.detect",
        "arm.grab",
        "arm.place",
    ]


def test_happy_path_continue() -> None:
    plan = handcrafted_plan()
    registry = SkillRegistry()
    registry.register(ArmGrabMockSkill(["grabbed"]))
    executor = Executor(
        registry=registry,
        planner=StaticPlanMockPlanner(plan),
        feedback_evaluator=SequenceFeedbackEvaluator([FeedbackDecision.CONTINUE]),
        image_fn=mock_image_fn,
    )

    result = executor.run("pick up the cube", plan)
    assert result.status == "done"
    assert result.replans == 0


def test_replan_path() -> None:
    plan = handcrafted_plan()
    registry = SkillRegistry()
    registry.register(ArmGrabMockSkill(["failed", "grabbed"]))
    executor = Executor(
        registry=registry,
        planner=StaticPlanMockPlanner(plan),
        feedback_evaluator=SequenceFeedbackEvaluator(
            [FeedbackDecision.REPLAN, FeedbackDecision.CONTINUE]
        ),
        image_fn=mock_image_fn,
        max_replans=1,
    )

    result = executor.run("pick up the cube", plan)
    assert result.replans == 1
    assert result.status != "replan_limit"
