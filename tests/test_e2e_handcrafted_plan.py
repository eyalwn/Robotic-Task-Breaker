from __future__ import annotations

from core.plan import Executor, FeedbackDecision, Plan, validate_plan
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


class MockPlanner:
    def __init__(self, plan_to_return: Plan) -> None:
        self._plan_to_return = plan_to_return

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        return self._plan_to_return


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


def test_happy_path_continue() -> None:
    plan = handcrafted_plan()
    registry = SkillRegistry()
    registry.register(ArmGrabMockSkill(["grabbed"]))
    executor = Executor(
        registry=registry,
        planner=MockPlanner(plan),
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
        planner=MockPlanner(plan),
        feedback_evaluator=SequenceFeedbackEvaluator([FeedbackDecision.REPLAN, FeedbackDecision.CONTINUE]),
        image_fn=mock_image_fn,
        max_replans=1,
    )

    result = executor.run("pick up the cube", plan)
    assert result.replans == 1
    assert result.status != "replan_limit"
