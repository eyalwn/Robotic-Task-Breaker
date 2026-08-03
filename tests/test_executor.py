"""Executor control-flow tests with fake env and VLA."""
from __future__ import annotations

from unittest.mock import patch

from conftest import (
    FakeEnv,
    FakeVLA,
    MockSkill,
    build_executor,
    make_plan,
    run_plan,
)
from executor import BoundaryReason, Executor
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
    run_plan(executor, plan)
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
    run_plan(executor, plan)
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
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.FEEDBACK_POINT
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
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.EXECUTION_ERROR
    assert "while loop exceeded max_iterations=3" in (result.error or "")
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
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.PLAN_DONE
    assert result.return_value == "result"


def test_feedback_returns_boundary() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan({"kind": "feedback", "description": "all good"})
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.FEEDBACK_POINT
    assert result.feedback_description == "all good"
    assert result.image_bytes == b"main-image"


def test_feedback_discards_remaining_plan() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan(
        {
            "kind": "seq",
            "steps": [
                {"kind": "call", "skill": "mock.skill"},
                {"kind": "feedback", "description": "checkpoint"},
                {"kind": "call", "skill": "mock.skill"},
            ],
        }
    )
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.FEEDBACK_POINT
    assert len(skill.calls) == 1


def test_rerun_plan_restarts_from_scratch() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan(
        {
            "kind": "seq",
            "steps": [
                {"kind": "call", "skill": "mock.skill"},
                {"kind": "feedback", "description": "checkpoint"},
            ],
        }
    )
    first = run_plan(executor, plan)
    assert first.reason == BoundaryReason.FEEDBACK_POINT
    assert len(skill.calls) == 1

    second = executor.run(plan)
    assert second.reason == BoundaryReason.FEEDBACK_POINT
    assert len(skill.calls) == 2


def test_return_node() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan({"kind": "return", "value": {"kind": "literal", "value": 42}})
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.PLAN_DONE
    assert result.return_value == 42


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
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.STEP_BUDGET
    assert len(skill.calls) == 2


def test_feedback_appends_prior_skill_result() -> None:
    skill = MockSkill({"status": "stopped", "grasp_confirmed": False})
    registry = SkillRegistry()
    registry.register(skill)
    executor = Executor(
        registry=registry,
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
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.FEEDBACK_POINT
    assert result.feedback_prompt is not None
    assert "Prior skill result:" in result.feedback_prompt
    assert "grasp_confirmed" in result.feedback_prompt


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
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.PLAN_DONE
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
    result = run_plan(executor, plan)
    assert result.reason == BoundaryReason.PLAN_DONE
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
    run_plan(executor, plan)
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
