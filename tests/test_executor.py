"""Executor tests with fake env and VLA."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from conftest import FakeEnv, FakeVLA, build_executor, make_plan, run_plan
from executor import StopReason


def _step_key(skill: str, **fields: str) -> tuple[str, frozenset]:
    step = {"skill": skill, **fields}
    return skill, frozenset(step.items())


def test_steps_execute_in_order() -> None:
    grab = {"skill": "grab", "object_name": "a"}
    place = {
        "skill": "place",
        "object_name": "a",
        "target_name": "basket",
        "position": "inside",
    }
    close = {"skill": "close", "target_name": "drawer"}
    env = FakeEnv(
        subtask_after={
            ("grab", frozenset(grab.items())): 1,
            ("place", frozenset(place.items())): 2,
            ("close", frozenset(close.items())): 3,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([grab, place, close])
    with patch("config.MAX_SKILL_STEPS", 1):
        run_plan(executor, plan)
    assert len(vla.calls) == 3
    assert vla.calls[0][2] == "pick up the a"
    assert vla.calls[1][2] == "place the a inside the basket"
    assert vla.calls[2][2] == "close the drawer"


def test_runs_all_steps_without_early_stop() -> None:
    env = FakeEnv(
        subtask_satisfied={
            _step_key("grab", object_name="a"): True,
            _step_key(
                "place",
                object_name="a",
                target_name="basket",
                position="inside",
            ): True,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan(
        [
            {"skill": "grab", "object_name": "a"},
            {
                "skill": "place",
                "object_name": "a",
                "target_name": "basket",
                "position": "inside",
            },
        ]
    )
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(vla.calls) == 2
    assert len(env.step_calls) == 2


def test_subtask_timeout_returns_summary() -> None:
    env = FakeEnv()
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan(
        [
            {"skill": "grab", "object_name": "cube"},
            {"skill": "close", "target_name": "drawer"},
        ]
    )
    with patch("config.MAX_SKILL_STEPS", 2):
        result = run_plan(executor, plan)
    assert result.reason == StopReason.SUBTASK_TIMEOUT
    assert len(env.step_calls) == 2


def test_grab_subtask_success_immediate() -> None:
    env = FakeEnv(subtask_satisfied={_step_key("grab", object_name="alphabet_soup_1"): True})
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "grab", "object_name": "alphabet_soup_1"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_grab_calls_vla_and_succeeds() -> None:
    env = FakeEnv(
        subtask_after={_step_key("grab", object_name="alphabet_soup_1"): 2}
    )
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "grab", "object_name": "alphabet_soup_1"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(vla.calls) == 1
    assert vla.calls[0][2] == "pick up the alphabet_soup_1"
    assert len(env.step_calls) == 2


def test_skill_timeout_contract() -> None:
    """Skills without subtask success time out at the per-skill cap."""
    env = FakeEnv()
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "grab", "object_name": "cube_1"}
    with patch("config.MAX_SKILL_STEPS", 3):
        assert executor._run_skill_successfuly(step) is False
    assert len(env.step_calls) == 3


def test_skill_done_when_subtask_already_satisfied() -> None:
    env = FakeEnv(subtask_satisfied={_step_key("grab", object_name="cube_1"): True})
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "grab", "object_name": "cube_1"}
    assert executor._run_skill_successfuly(step) is True
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_place_subtask_success_ends_early() -> None:
    env = FakeEnv(
        subtask_satisfied={
            _step_key(
                "place",
                object_name="cream_cheese_1",
                target_name="basket_1_contain_region",
                position="inside",
            ): True,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {
        "skill": "place",
        "object_name": "cream_cheese_1",
        "target_name": "basket_1_contain_region",
        "position": "inside",
    }
    assert executor._run_skill_successfuly(step) is True
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_place_subtask_success_after_steps() -> None:
    env = FakeEnv(
        subtask_after={
            _step_key(
                "place",
                object_name="alphabet_soup_1",
                target_name="basket_1_contain_region",
                position="inside",
            ): 1,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {
        "skill": "place",
        "object_name": "alphabet_soup_1",
        "target_name": "basket_1_contain_region",
        "position": "inside",
    }
    assert executor._run_skill_successfuly(step) is True
    assert len(vla.calls) == 1
    assert vla.calls[0][2] == "place the alphabet_soup_1 inside the basket_1_contain_region"
    assert len(env.step_calls) == 2


@pytest.mark.parametrize(
    ("skill", "step", "expected_task"),
    [
        (
            "grab",
            {"skill": "grab", "object_name": "bowl_1"},
            "pick up the bowl_1",
        ),
        (
            "place",
            {
                "skill": "place",
                "object_name": "bowl_1",
                "target_name": "plate",
                "position": "on",
            },
            "place the bowl_1 on the plate",
        ),
        (
            "close",
            {"skill": "close", "target_name": "microwave_1"},
            "close the microwave_1",
        ),
        (
            "turn_on",
            {"skill": "turn_on", "target_name": "stove_1"},
            "turn on the stove_1",
        ),
    ],
)
def test_skill_task_strings(skill: str, step: dict, expected_task: str) -> None:
    env = FakeEnv(subtask_after={(skill, frozenset(step.items())): 1})
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    run_plan(executor, [step])
    assert vla.calls[0][2] == expected_task

def test_unknown_skill_raises() -> None:
    env = FakeEnv()
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    result = run_plan(executor, [{"skill": "transfer", "object_name": "a"}])
    assert result.reason == StopReason.EXECUTION_ERROR


def test_executes_full_returned_chunk() -> None:
    """Executor must not trim actions; it runs every action from the service."""
    chunk_size = 7
    actions = [[float(i)] * 7 for i in range(chunk_size)]
    env = FakeEnv()
    vla = FakeVLA(actions=actions)
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "grab", "object_name": "cube"}
    with patch("config.MAX_SKILL_STEPS", chunk_size):
        assert executor._run_skill_successfuly(step) is False
    assert len(vla.calls) == 1
    assert len(env.step_calls) == chunk_size
    assert env.step_calls[-1][0] == float(chunk_size - 1)
