from __future__ import annotations

import pytest

from core.plan import (
    FeedbackDecision,
    FeedbackOutcome,
    Executor,
    Plan,
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
    def __init__(self, decision: FeedbackDecision, reason: str = "") -> None:
        self.decision = decision
        self.reason = reason
        self.last_description: str | None = None

    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackOutcome:
        self.last_description = description
        return FeedbackOutcome(decision=self.decision, reason=self.reason)


def mock_image_fn() -> bytes:
    return b"fake-image"


def build_executor(
    skill: Skill,
    decision: FeedbackDecision = FeedbackDecision.CONTINUE,
    step_budget: int = 30,
) -> Executor:
    registry = SkillRegistry()
    registry.register(skill)
    return Executor(
        registry=registry,
        feedback_evaluator=MockFeedbackEvaluator(decision),
        image_fn=mock_image_fn,
        goal="test-goal",
        step_budget=step_budget,
    )


def test_seq_executes_in_order() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
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
    executor.run(plan)
    assert [call["x"] for call in skill.calls] == [1, 2, 3]


def test_if_then() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
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
    executor.run(plan)
    assert len(skill.calls) == 1


def test_if_else() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
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
    executor.run(plan)
    assert len(skill.calls) == 0


def test_while_max_iterations() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
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
        executor.run(plan)
    assert len(skill.calls) == 3


def test_call_assigns_return() -> None:
    skill = MockSkill("result")
    executor = build_executor(skill)
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
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value == "result"


def test_feedback_continue() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, decision=FeedbackDecision.CONTINUE)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {"kind": "feedback", "description": "all good"},
        }
    )
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value is None


def test_feedback_stop() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, decision=FeedbackDecision.STOP)
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "halt"}})
    result = executor.run(plan)
    assert result.status == "stopped"


def test_feedback_replan() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, decision=FeedbackDecision.REPLAN)
    plan = Plan.model_validate({"goal": "g", "body": {"kind": "feedback", "description": "retry"}})
    result = executor.run(plan)
    assert result.status == "replan"


def test_return_node() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {"kind": "return", "value": {"kind": "literal", "value": 42}},
        }
    )
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value == 42


def test_step_budget() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, step_budget=2)
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
    result = executor.run(plan)
    assert result.status == "step_budget"
    assert len(skill.calls) == 2


def test_var_unbound_raises() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = Plan.model_validate(
        {"goal": "g", "body": {"kind": "return", "value": {"kind": "var", "name": "missing"}}}
    )
    with pytest.raises(NameError):
        executor.run(plan)


def test_binop_equality() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
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

    true_result = executor.run(true_plan)
    false_result = executor.run(false_plan)
    assert true_result.status == "done"
    assert false_result.status == "done"
    assert true_result.value is True
    assert false_result.value is False


def test_binop_and_short_circuit() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
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
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value is False


def test_binop_in_true() -> None:
    skill = MockSkill(["apple", "banana"])
    executor = build_executor(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill", "args": {}, "assign_to": "objects"},
                    {
                        "kind": "if",
                        "condition": {
                            "kind": "binop",
                            "op": "in",
                            "left": {"kind": "literal", "value": "apple"},
                            "right": {"kind": "var", "name": "objects"},
                        },
                        "then_branch": {"kind": "call", "skill": "mock.skill", "args": {}},
                    },
                ],
            },
        }
    )
    executor.run(plan)
    assert len(skill.calls) == 2


def test_binop_in_false() -> None:
    skill = MockSkill(["apple", "banana"])
    executor = build_executor(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill", "args": {}, "assign_to": "objects"},
                    {
                        "kind": "if",
                        "condition": {
                            "kind": "binop",
                            "op": "in",
                            "left": {"kind": "literal", "value": "grape"},
                            "right": {"kind": "var", "name": "objects"},
                        },
                        "then_branch": {"kind": "call", "skill": "mock.skill", "args": {}},
                    },
                ],
            },
        }
    )
    executor.run(plan)
    assert len(skill.calls) == 1


def test_binop_in_none_right_returns_false() -> None:
    skill = MockSkill(None)
    executor = build_executor(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "call", "skill": "mock.skill", "args": {}, "assign_to": "objects"},
                    {
                        "kind": "if",
                        "condition": {
                            "kind": "binop",
                            "op": "in",
                            "left": {"kind": "literal", "value": "apple"},
                            "right": {"kind": "var", "name": "objects"},
                        },
                        "then_branch": {"kind": "call", "skill": "mock.skill", "args": {}},
                    },
                ],
            },
        }
    )
    executor.run(plan)
    assert len(skill.calls) == 1


def test_not_expr() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "return",
                "value": {"kind": "not", "operand": {"kind": "literal", "value": True}},
            },
        }
    )
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value is False


def test_feedback_continue_then_return() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, decision=FeedbackDecision.CONTINUE)
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {"kind": "feedback", "description": "all good"},
                    {"kind": "return", "value": {"kind": "literal", "value": "x"}},
                ],
            },
        }
    )
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value == "x"


def test_feedback_trace_includes_reason(tmp_path) -> None:
    import json

    from core.plan.trace import TraceWriter

    skill = MockSkill("ok")
    registry = SkillRegistry()
    registry.register(skill)
    trace = TraceWriter(tmp_path, "trace-reason")
    executor = Executor(
        registry=registry,
        feedback_evaluator=MockFeedbackEvaluator(
            FeedbackDecision.CONTINUE, reason="object held"
        ),
        image_fn=mock_image_fn,
        goal="test-goal",
        trace=trace,
    )
    plan = Plan.model_validate(
        {"goal": "g", "body": {"kind": "feedback", "description": "grasp check"}}
    )
    executor.run(plan)
    trace.close()
    event = json.loads(trace.path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert event["event"] == "feedback"
    assert event["decision"] == "continue"
    assert event["reason"] == "object held"


def test_feedback_appends_prior_skill_result() -> None:
    skill = MockSkill('{"status":"stopped","grasp_confirmed":false}')
    feedback = MockFeedbackEvaluator(FeedbackDecision.CONTINUE)
    registry = SkillRegistry()
    registry.register(skill)
    executor = Executor(
        registry=registry,
        feedback_evaluator=feedback,
        image_fn=mock_image_fn,
        goal="test-goal",
    )
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "seq",
                "steps": [
                    {
                        "kind": "call",
                        "skill": "mock.skill",
                        "args": {},
                    },
                    {"kind": "feedback", "description": "Is the object grasped?"},
                ],
            },
        }
    )
    executor.run(plan)
    assert feedback.last_description is not None
    assert "Prior skill result:" in feedback.last_description
    assert "grasp_confirmed" in feedback.last_description


def test_executor_resolves_skill_alias() -> None:
    target = MockSkill(["obj_a"])
    registry = SkillRegistry()
    registry.register(target)
    registry.register_alias("list_objects", target.name)
    executor = Executor(
        registry=registry,
        feedback_evaluator=MockFeedbackEvaluator(FeedbackDecision.CONTINUE),
        image_fn=mock_image_fn,
        goal="test-goal",
    )
    plan = Plan.model_validate(
        {
            "goal": "g",
            "body": {
                "kind": "call",
                "skill": "list_objects",
                "assign_to": "objects",
                "args": {},
            },
        }
    )
    executor.run(plan)
    assert target.calls == [{}]
