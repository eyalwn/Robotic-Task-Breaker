from __future__ import annotations

import pytest

from core.plan import (
    FeedbackDecision,
    Interpreter,
    Plan,
    ReplanSignal,
    ReturnSignal,
    StepBudgetExceeded,
    StopSignal,
)
from core.skills import Skill, SkillRegistry


class MockSkill(Skill):
    name = "mock.skill"
    description = "mock skill"
    parameters = []
    returns = "Any"

    def __init__(self, return_value: object = None) -> None:
        self.return_value = return_value
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.return_value


class MockFeedbackEvaluator:
    def __init__(self, decision: FeedbackDecision) -> None:
        self.decision = decision

    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackDecision:
        return self.decision


def mock_image_fn() -> bytes:
    return b"fake-image"


def build_interpreter(
    skill: Skill,
    decision: FeedbackDecision = FeedbackDecision.CONTINUE,
    step_budget: int = 30,
) -> Interpreter:
    registry = SkillRegistry()
    registry.register(skill)
    return Interpreter(
        registry=registry,
        feedback_evaluator=MockFeedbackEvaluator(decision),
        image_fn=mock_image_fn,
        goal="test-goal",
        step_budget=step_budget,
    )


def test_seq_executes_in_order() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill", "args": {"x": {"kind": "literal", "value": 1}}},
                    {"kind": "call", "skill": "mock.skill", "args": {"x": {"kind": "literal", "value": 2}}},
                    {"kind": "call", "skill": "mock.skill", "args": {"x": {"kind": "literal", "value": 3}}},
                ],
            },
        }
    )
    interp.run(plan)
    assert [call["x"] for call in skill.calls] == [1, 2, 3]


def test_if_then() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "if",
                "condition": {"kind": "literal", "value": True},
                "then_branch": {"kind": "call", "skill": "mock.skill"},
                "else_branch": {"kind": "feedback", "description": "skip"},
            },
        }
    )
    interp.run(plan)
    assert len(skill.calls) == 1


def test_if_else() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "if",
                "condition": {"kind": "literal", "value": False},
                "then_branch": {"kind": "call", "skill": "mock.skill"},
                "else_branch": {"kind": "feedback", "description": "else path"},
            },
        }
    )
    interp.run(plan)
    assert len(skill.calls) == 0


def test_while_max_iterations() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "while",
                "condition": {"kind": "literal", "value": True},
                "max_iterations": 3,
                "body": {"kind": "call", "skill": "mock.skill"},
            },
        }
    )
    with pytest.raises(RuntimeError, match="while loop exceeded max_iterations=3"):
        interp.run(plan)
    assert len(skill.calls) == 3


def test_call_assigns_return() -> None:
    skill = MockSkill("result")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill", "assign_to": "x"},
                    {"kind": "return", "value": {"kind": "var", "name": "x"}},
                ],
            },
        }
    )
    with pytest.raises(ReturnSignal) as exc_info:
        interp.run(plan)
    assert exc_info.value.value == "result"


def test_feedback_continue() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill, decision=FeedbackDecision.CONTINUE)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {"kind": "feedback", "description": "all good"},
        }
    )
    assert interp.run(plan) is None


def test_feedback_stop() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill, decision=FeedbackDecision.STOP)
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "halt"}})
    with pytest.raises(StopSignal) as exc_info:
        interp.run(plan)
    assert not hasattr(exc_info.value, "value")


def test_feedback_replan() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill, decision=FeedbackDecision.REPLAN)
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry"}})
    with pytest.raises(ReplanSignal):
        interp.run(plan)


def test_return_node() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {"kind": "return", "value": {"kind": "literal", "value": 42}},
        }
    )
    with pytest.raises(ReturnSignal) as exc_info:
        interp.run(plan)
    assert exc_info.value.value == 42


def test_step_budget() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill, step_budget=2)
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
    with pytest.raises(StepBudgetExceeded):
        interp.run(plan)
    assert len(skill.calls) == 2


def test_var_unbound_raises() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {"goal": "g", "body": {"kind": "return", "value": {"kind": "var", "name": "missing"}}}
    )
    with pytest.raises(NameError):
        interp.run(plan)


def test_binop_equality() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    true_plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "return",
                "value": {
                    "kind": "binop",
                    "op": "==",
                    "left": {"kind": "literal", "value": 1},
                    "right": {"kind": "literal", "value": 1},
                },
            },
        }
    )
    false_plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "return",
                "value": {
                    "kind": "binop",
                    "op": "==",
                    "left": {"kind": "literal", "value": 1},
                    "right": {"kind": "literal", "value": 2},
                },
            },
        }
    )

    with pytest.raises(ReturnSignal) as true_sig:
        interp.run(true_plan)
    with pytest.raises(ReturnSignal) as false_sig:
        interp.run(false_plan)
    assert true_sig.value.value is True
    assert false_sig.value.value is False


def test_binop_and_short_circuit() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "return",
                "value": {
                    "kind": "binop",
                    "op": "and",
                    "left": {"kind": "literal", "value": False},
                    "right": {"kind": "var", "name": "missing"},
                },
            },
        }
    )
    with pytest.raises(ReturnSignal) as sig:
        interp.run(plan)
    assert sig.value.value is False


def test_not_expr() -> None:
    skill = MockSkill("ok")
    interp = build_interpreter(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "return",
                "value": {"kind": "not", "operand": {"kind": "literal", "value": True}},
            },
        }
    )
    with pytest.raises(ReturnSignal) as sig:
        interp.run(plan)
    assert sig.value.value is False
