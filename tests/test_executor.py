"""Executor control-flow tests with fake env and VLA."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from conftest import (
    FakeEnv,
    FakeVLA,
    MockSkill,
    build_executor,
    make_plan,
    run_plan,
)
from executor import StopReason


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
    assert result.reason == StopReason.FEEDBACK_POINT
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
    assert result.reason == StopReason.EXECUTION_ERROR
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
    assert result.reason == StopReason.PLAN_DONE
    assert result.return_value == "result"


def test_feedback_returns_summary() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan({"kind": "feedback", "description": "all good"})
    result = run_plan(executor, plan)
    assert result.reason == StopReason.FEEDBACK_POINT
    assert result.feedback_description == "all good"
    assert result.feedback_prompt == "all good"


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
    assert result.reason == StopReason.FEEDBACK_POINT
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
    assert first.reason == StopReason.FEEDBACK_POINT
    assert len(skill.calls) == 1

    second = executor.run(plan)
    assert second.reason == StopReason.FEEDBACK_POINT
    assert len(skill.calls) == 2


def test_return_node() -> None:
    skill = MockSkill("ok")
    executor = build_executor(skill)
    plan = make_plan({"kind": "return", "value": {"kind": "literal", "value": 42}})
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert result.return_value == 42


def test_step_budget_from_env_steps() -> None:
    """Episode budget exhaustion stops execution when env steps are consumed."""
    env = FakeEnv(succeed_after=None, max_steps=2)
    vla = FakeVLA(actions=[[0.0] * 7])
    skill = MockSkill("unused")
    executor = build_executor(skill, env=env, vla=vla)
    plan = make_plan(
        {
            "kind": "seq",
            "steps": [
                {
                    "kind": "call",
                    "skill": "arm.grab",
                    "args": {"object_name": {"kind": "literal", "value": "cube_1"}},
                },
                {"kind": "call", "skill": "mock.skill"},
            ],
        }
    )
    result = run_plan(executor, plan)
    assert result.reason == StopReason.STEP_BUDGET
    assert len(env.step_calls) == 2
    assert len(skill.calls) == 0


def test_feedback_appends_prior_skill_result() -> None:
    skill = MockSkill({"status": "stopped", "grasp_confirmed": False})
    executor = build_executor(skill)
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
    assert result.reason == StopReason.FEEDBACK_POINT
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
    assert result.reason == StopReason.PLAN_DONE
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
    assert result.reason == StopReason.PLAN_DONE
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


def test_arm_skill_attempt_contract() -> None:
    """Arm skills return attempt-only payload, never status=failed from BDDL."""
    env = FakeEnv(succeed_after=None)
    vla = FakeVLA(actions=[[0.0] * 7])
    skill = MockSkill("unused")
    executor = build_executor(skill, env=env, vla=vla)
    with patch("config.MAX_SKILL_STEPS", 3):
        outcome = executor._run_arm_skill(
            "arm.grab", {"object_name": "cube_1"}
        )
    assert outcome == {"status": "attempted", "env_steps": 3, "task_success": False}
    assert len(env.step_calls) == 3


def test_arm_skill_attempt_when_task_already_solved() -> None:
    env = FakeEnv(succeed_after=0)
    vla = FakeVLA()
    skill = MockSkill("unused")
    executor = build_executor(skill, env=env, vla=vla)
    outcome = executor._run_arm_skill("arm.grab", {"object_name": "cube_1"})
    assert outcome == {"status": "attempted", "env_steps": 0, "task_success": True}
    assert vla.calls == []


@pytest.mark.parametrize(
    ("skill", "args", "expected_task"),
    [
        (
            "arm.transfer",
            {"object_name": "bowl_1", "location": "plate"},
            "move the bowl to the plate",
        ),
        (
            "arm.open",
            {"target_name": "top_drawer_1"},
            "open the top drawer",
        ),
        (
            "arm.close",
            {"target_name": "microwave_1"},
            "close the microwave",
        ),
        (
            "arm.push",
            {"target_name": "mug_1", "destination_or_direction": "left"},
            "push the mug to the left",
        ),
        (
            "arm.pull",
            {"target_name": "mug_1", "destination_or_direction": "left"},
            "pull the mug to the left",
        ),
    ],
)
def test_arm_skill_task_strings(skill: str, args: dict, expected_task: str) -> None:
    env = FakeEnv(succeed_after=1)
    vla = FakeVLA()
    skill_obj = MockSkill("unused")
    executor = build_executor(skill_obj, env=env, vla=vla)
    plan = make_plan(
        {
            "kind": "call",
            "skill": skill,
            "args": {
                key: {"kind": "literal", "value": value} for key, value in args.items()
            },
        }
    )
    run_plan(executor, plan)
    assert vla.calls[0][3] == expected_task


@pytest.mark.parametrize(
    ("skill", "args", "missing"),
    [
        ("arm.grab", {}, "object_name"),
        ("arm.place", {"object_name": "cube"}, "location"),
        ("arm.transfer", {"object_name": "bowl"}, "location"),
        ("arm.open", {}, "target_name"),
        ("arm.close", {}, "target_name"),
        ("arm.push", {"target_name": "mug"}, "destination_or_direction"),
        ("arm.pull", {"target_name": "mug"}, "destination_or_direction"),
    ],
)
def test_arm_skill_requires_args(skill: str, args: dict, missing: str) -> None:
    env = FakeEnv()
    vla = FakeVLA()
    skill_obj = MockSkill("unused")
    executor = build_executor(skill_obj, env=env, vla=vla)
    with pytest.raises(ValueError, match=f"{skill} requires {missing!r}"):
        executor._run_arm_skill(skill, args)
