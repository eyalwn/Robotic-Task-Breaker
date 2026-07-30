"""Executor control-flow tests with fake env and VLA."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from conftest import (
    FakeEnv,
    FakeVLA,
    MockFeedbackEvaluator,
    MockSkill,
    build_executor,
    make_plan,
    mock_image_fn,
)
from executor import Executor, FeedbackDecision
from skills import SkillRegistry


def test_seq_executes_in_order() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan(
        {
            "kind": "seq",
            "steps": [
                {"kind": "call", "skill": "mock.skill", "args": {"x": {"kind": "literal", "value": 1}}},
                {"kind": "call", "skill": "mock.skill", "args": {"x": {"kind": "literal", "value": 2}}},
                {"kind": "call", "skill": "mock.skill", "args": {"x": {"kind": "literal", "value": 3}}},
            ],
        }
    )
    executor.run(plan)
    assert [call["x"] for call in skill.calls] == [1, 2, 3]


def test_if_then() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan(
        {
            "kind": "if",
            "condition": {"kind": "literal", "value": True},
            "then_branch": {"kind": "call", "skill": "mock.skill"},
            "else_branch": {"kind": "feedback", "description": "skip"},
        }
    )
    executor.run(plan)
    assert len(skill.calls) == 1


def test_if_else() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan(
        {
            "kind": "if",
            "condition": {"kind": "literal", "value": False},
            "then_branch": {"kind": "call", "skill": "mock.skill"},
            "else_branch": {"kind": "feedback", "description": "else path"},
        }
    )
    executor.run(plan)
    assert len(skill.calls) == 0


def test_while_max_iterations() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan(
        {
            "kind": "while",
            "condition": {"kind": "literal", "value": True},
            "max_iterations": 3,
            "body": {"kind": "call", "skill": "mock.skill"},
        }
    )
    with pytest.raises(RuntimeError, match="while loop exceeded max_iterations=3"):
        executor.run(plan)
    assert len(skill.calls) == 3


def test_call_assigns_return() -> None:
    skill = MockSkill("result")
    executor = build_executor(skill)
    plan = make_plan(
        {
            "kind": "seq",
            "steps": [
                {"kind": "call", "skill": "mock.skill", "assign_to": "x"},
                {"kind": "return", "value": {"kind": "var", "name": "x"}},
            ],
        }
    )
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value == "result"


def test_feedback_continue() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, decision=FeedbackDecision.CONTINUE)
    plan = make_plan({"kind": "feedback", "description": "all good"})
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value is None


def test_feedback_stop() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, decision=FeedbackDecision.STOP)
    plan = make_plan({"kind": "feedback", "description": "halt"})
    result = executor.run(plan)
    assert result.status == "stopped"


def test_feedback_replan() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, decision=FeedbackDecision.REPLAN)
    plan = make_plan({"kind": "feedback", "description": "retry"})
    result = executor.run(plan)
    assert result.status == "replan"


def test_return_node() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan({"kind": "return", "value": {"kind": "literal", "value": 42}})
    result = executor.run(plan)
    assert result.status == "done"
    assert result.value == 42


def test_step_budget() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill, step_budget=2)
    plan = make_plan(
        {
            "kind": "seq",
            "steps": [
                {"kind": "call", "skill": "mock.skill"},
                {"kind": "call", "skill": "mock.skill"},
                {"kind": "call", "skill": "mock.skill"},
            ],
        }
    )
    result = executor.run(plan)
    assert result.status == "step_budget"
    assert len(skill.calls) == 2


def test_feedback_appends_prior_skill_result() -> None:
    skill = MockSkill({"status": "stopped", "grasp_confirmed": False})
    feedback = MockFeedbackEvaluator(FeedbackDecision.CONTINUE)
    registry = SkillRegistry()
    registry.register(skill)
    executor = Executor(
        registry=registry,
        feedback_evaluator=feedback,
        image_fn=mock_image_fn,
        goal="test-goal",
        env=FakeEnv(),
        vla=FakeVLA(),
    )
    plan = make_plan(
        {
            "kind": "seq",
            "steps": [
                {"kind": "call", "skill": "mock.skill", "args": {}},
                {"kind": "feedback", "description": "Is the object grasped?"},
            ],
        }
    )
    executor.run(plan)
    assert feedback.last_description is not None
    assert "Prior skill result:" in feedback.last_description
    assert "grasp_confirmed" in feedback.last_description


def test_arm_grab_success_immediate() -> None:
    env = FakeEnv(succeed_after=0)
    vla = FakeVLA()
    skill = MockSkill("unused")
    executor = build_executor(skill, env=env, vla=vla)
    plan = make_plan(
        {
            "kind": "call",
            "skill": "arm.grab",
            "args": {"object_name": {"kind": "literal", "value": "alphabet_soup_1"}},
            "assign_to": "outcome",
        }
    )
    result = executor.run(plan)
    assert result.status == "done"
    assert vla.calls == []
    assert env.step_calls == []


def test_arm_grab_calls_vla_and_succeeds() -> None:
    env = FakeEnv(succeed_after=2)
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7])
    skill = MockSkill("unused")
    executor = build_executor(skill, env=env, vla=vla)
    plan = make_plan(
        {
            "kind": "call",
            "skill": "arm.grab",
            "args": {"object_name": {"kind": "literal", "value": "alphabet_soup_1"}},
        }
    )
    result = executor.run(plan)
    assert result.status == "done"
    assert len(vla.calls) == 1
    assert vla.calls[0][3] == "pick up the alphabet soup"
    assert len(env.step_calls) == 2


def test_arm_place_task_string() -> None:
    env = FakeEnv(succeed_after=1)
    vla = FakeVLA()
    skill = MockSkill("unused")
    executor = build_executor(skill, env=env, vla=vla)
    plan = make_plan(
        {
            "kind": "call",
            "skill": "arm.place",
            "args": {
                "object_name": {"kind": "literal", "value": "cream_cheese_1"},
                "location": {"kind": "literal", "value": "basket"},
            },
        }
    )
    executor.run(plan)
    assert vla.calls[0][3] == "put the cream cheese in the basket"


def test_manipulate_failure_contract() -> None:
    env = FakeEnv(succeed_after=None)
    vla = FakeVLA(actions=[[0.0] * 7])
    skill = MockSkill("unused")
    executor = build_executor(skill, env=env, vla=vla)
    with patch("config.MAX_SKILL_STEPS", 3):
        outcome = executor._manipulate("arm.grab", "cube_1")
    assert outcome == {"status": "failed", "steps": 3}
    assert len(env.step_calls) == 3
