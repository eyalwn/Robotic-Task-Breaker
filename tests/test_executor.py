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
    step_a = {"skill": "transfer", "object_name": "a", "target_name": "basket"}
    step_b = {"skill": "transfer", "object_name": "b", "target_name": "plate_seen"}
    step_c = {"skill": "transfer", "object_name": "c", "target_name": "bin"}
    env = FakeEnv(
        subtask_after={
            ("transfer", frozenset(step_a.items())): 1,
            ("transfer", frozenset(step_b.items())): 2,
            ("transfer", frozenset(step_c.items())): 3,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([step_a, step_b, step_c])
    with patch("config.MAX_SKILL_STEPS", 1):
        run_plan(executor, plan)
    assert len(vla.calls) == 3
    assert vla.calls[0][2] == "Put the a into the basket"
    assert vla.calls[1][2] == "Put the b into the plate_seen"
    assert vla.calls[2][2] == "Put the c into the bin"


def test_runs_all_steps_without_early_stop() -> None:
    env = FakeEnv(
        subtask_satisfied={
            _step_key("transfer", object_name="a", target_name="basket"): True,
            _step_key(
                "transfer",
                object_name="b",
                target_name="plate_seen",
            ): True,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan(
        [
            {"skill": "transfer", "object_name": "a", "target_name": "basket"},
            {"skill": "transfer", "object_name": "b", "target_name": "plate_seen"},
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
            {"skill": "transfer", "object_name": "cube", "target_name": "drawer"},
        ]
    )
    with patch("config.MAX_SKILL_STEPS", 2):
        result = run_plan(executor, plan)
    assert result.reason == StopReason.SUBTASK_TIMEOUT
    assert len(env.step_calls) == 2


def test_transfer_subtask_success_immediate() -> None:
    env = FakeEnv(
        subtask_satisfied={
            _step_key(
                "transfer",
                object_name="alphabet_soup_1",
                target_name="basket_1_contain_region",
            ): True
        }
    )
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    plan = make_plan(
        [
            {
                "skill": "transfer",
                "object_name": "alphabet_soup_1",
                "target_name": "basket_1_contain_region",
            }
        ]
    )
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_transfer_calls_vla_and_succeeds() -> None:
    env = FakeEnv(
        subtask_after={
            _step_key(
                "transfer",
                object_name="alphabet_soup_1",
                target_name="basket_1_contain_region",
            ): 2
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan(
        [
            {
                "skill": "transfer",
                "object_name": "alphabet_soup_1",
                "target_name": "basket_1_contain_region",
            }
        ]
    )
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(vla.calls) == 1
    assert (
        vla.calls[0][2]
        == "Put the alphabet_soup_1 into the basket_1_contain_region"
    )
    assert len(env.step_calls) == 2


def test_skill_timeout_contract() -> None:
    """Skills without subtask success time out at the per-skill cap."""
    env = FakeEnv()
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "transfer", "object_name": "cube_1", "target_name": "basket"}
    with patch("config.MAX_SKILL_STEPS", 3):
        assert executor._run_step_successfuly(step) is False
    assert len(env.step_calls) == 3


def test_skill_done_when_subtask_already_satisfied() -> None:
    env = FakeEnv(
        subtask_satisfied={
            _step_key("transfer", object_name="cube_1", target_name="basket"): True
        }
    )
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "transfer", "object_name": "cube_1", "target_name": "basket"}
    assert executor._run_step_successfuly(step) is True
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_transfer_subtask_success_ends_early() -> None:
    env = FakeEnv(
        subtask_satisfied={
            _step_key(
                "transfer",
                object_name="cream_cheese_1",
                target_name="basket_1_contain_region",
            ): True,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {
        "skill": "transfer",
        "object_name": "cream_cheese_1",
        "target_name": "basket_1_contain_region",
    }
    assert executor._run_step_successfuly(step) is True
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_transfer_subtask_success_after_steps() -> None:
    env = FakeEnv(
        subtask_after={
            _step_key(
                "transfer",
                object_name="alphabet_soup_1",
                target_name="basket_1_contain_region",
            ): 1,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {
        "skill": "transfer",
        "object_name": "alphabet_soup_1",
        "target_name": "basket_1_contain_region",
    }
    assert executor._run_step_successfuly(step) is True
    assert len(vla.calls) == 1
    assert (
        vla.calls[0][2]
        == "Put the alphabet_soup_1 into the basket_1_contain_region"
    )
    assert len(env.step_calls) == 2


def test_transfer_task_string_is_checkpoint_native() -> None:
    step = {"skill": "transfer", "object_name": "banana", "target_name": "plate_seen"}
    env = FakeEnv(subtask_after={("transfer", frozenset(step.items())): 1})
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    run_plan(executor, [step])
    assert vla.calls[0][2] == "Put the banana into the plate_seen"


def test_unknown_skill_raises() -> None:
    env = FakeEnv()
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    result = run_plan(
        executor,
        [{"skill": "close", "target_name": "drawer"}],
    )
    assert result.reason == StopReason.EXECUTION_ERROR


def test_executes_full_returned_chunk() -> None:
    """Executor must not trim actions; it runs every action from the service."""
    chunk_size = 7
    actions = [[float(i)] * 7 for i in range(chunk_size)]
    env = FakeEnv()
    vla = FakeVLA(actions=actions)
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "transfer", "object_name": "cube", "target_name": "basket"}
    with patch("config.MAX_SKILL_STEPS", chunk_size):
        assert executor._run_step_successfuly(step) is False
    assert len(vla.calls) == 1
    assert len(env.step_calls) == chunk_size
    assert env.step_calls[-1][0] == float(chunk_size - 1)
