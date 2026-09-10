"""Executor tests with fake env and VLA."""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest

import config
from conftest import FakeEnv, FakeVLA, build_executor, make_plan, run_plan
from executor import SkillAbort, SkillTimeout, StopReason


class FakeEnvStepDone(FakeEnv):
    """Returns True from step() once the step count reaches done_after."""

    def __init__(self, done_after: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._done_after = done_after

    def step(self, action: list[float]) -> bool:
        super().step(action)
        return len(self.step_calls) >= self._done_after


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
    with patch("config.MAX_SKILL_STEPS", {"transfer": 1}):
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
    with patch("config.MAX_SKILL_STEPS", {"transfer": 2}):
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
    with patch("config.MAX_SKILL_STEPS", {"transfer": 3}):
        with pytest.raises(SkillTimeout, match="exceeded 3 steps"):
            executor._run_step(step)
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
    executor._run_step(step)
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
    executor._run_step(step)
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_transfer_subtask_success_after_steps() -> None:
    env = FakeEnv(
        subtask_after={
            _step_key(
                "transfer",
                object_name="alphabet_soup_1",
                target_name="basket_1_contain_region",
            ): 2,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {
        "skill": "transfer",
        "object_name": "alphabet_soup_1",
        "target_name": "basket_1_contain_region",
    }
    executor._run_step(step)
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
    with patch("config.MAX_SKILL_STEPS", {"transfer": chunk_size}):
        with pytest.raises(SkillTimeout, match="exceeded"):
            executor._run_step(step)
    assert len(vla.calls) == 1
    assert len(env.step_calls) == chunk_size
    assert env.step_calls[-1][0] == float(chunk_size - 1)


def test_env_step_done_mid_chunk_stops_remaining_actions() -> None:
    """env.step() returning True ends the skill before later chunk actions."""
    chunk_size = 7
    done_after = 2
    actions = [[float(i)] * 7 for i in range(chunk_size)]
    env = FakeEnvStepDone(done_after=done_after)
    vla = FakeVLA(actions=actions)
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "transfer", "object_name": "cube", "target_name": "basket"}
    with patch("config.MAX_SKILL_STEPS", {"transfer": chunk_size}):
        executor._run_step(step)
    assert len(vla.calls) == 1
    assert len(env.step_calls) == done_after
    assert env.step_calls[0][0] == 0.0
    assert env.step_calls[1][0] == 1.0


def test_env_step_done_reports_plan_done() -> None:
    env = FakeEnvStepDone(done_after=1)
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7, [0.2] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan(
        [{"skill": "transfer", "object_name": "cube", "target_name": "basket"}]
    )
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(env.step_calls) == 1


_TRANSFER = {"skill": "transfer", "object_name": "cube", "target_name": "basket"}


def test_approach_stall_raises() -> None:
    executor = build_executor()
    tracker = (False, math.inf, 0)
    for _ in range(config.SKILL_STALL_STEPS):
        tracker = executor._abort_if_stuck(_TRANSFER, tracker, (1.0, 1.0))
    with pytest.raises(SkillAbort, match="approaching object"):
        executor._abort_if_stuck(_TRANSFER, tracker, (1.0, 1.0))


def test_place_stall_after_hold_raises() -> None:
    executor = build_executor()
    tracker = executor._abort_if_stuck(
        _TRANSFER, (False, math.inf, 0), (0.05, 0.9)
    )
    assert tracker[0] is True
    for _ in range(config.SKILL_STALL_STEPS - 1):
        tracker = executor._abort_if_stuck(_TRANSFER, tracker, (0.05, 0.9))
    with pytest.raises(SkillAbort, match="moving to target"):
        executor._abort_if_stuck(_TRANSFER, tracker, (0.05, 0.9))


def test_drop_after_hold_raises() -> None:
    executor = build_executor()
    tracker = executor._abort_if_stuck(
        _TRANSFER, (False, math.inf, 0), (0.05, 0.9)
    )
    dropped = (config.TRANSFER_DROP_DISTANCE + 0.01, 0.9)
    with pytest.raises(SkillAbort, match="dropped"):
        executor._abort_if_stuck(_TRANSFER, tracker, dropped)


def test_improving_approach_does_not_stall() -> None:
    executor = build_executor()
    tracker = (False, math.inf, 0)
    for dist in (1.0, 0.8, 0.6, 0.4):
        tracker = executor._abort_if_stuck(_TRANSFER, tracker, (dist, 1.0))
    assert tracker == (False, 0.4, 0)


def test_approach_stall_run_is_skill_abort() -> None:
    env = FakeEnv()
    executor = build_executor(env=env, vla=FakeVLA(actions=[[0.0] * 7]))
    with patch("config.MAX_SKILL_STEPS", {"transfer": 250}):
        result = run_plan(executor, make_plan([_TRANSFER]))
    assert result.reason == StopReason.SKILL_ABORT
    assert "approaching object" in result.error
    assert len(env.step_calls) == config.SKILL_STALL_STEPS + 1


def test_place_stall_run_is_skill_abort() -> None:
    env = FakeEnv(distances=(0.05, 0.8))
    executor = build_executor(env=env, vla=FakeVLA(actions=[[0.0] * 7]))
    with patch("config.MAX_SKILL_STEPS", {"transfer": 250}):
        result = run_plan(executor, make_plan([_TRANSFER]))
    assert result.reason == StopReason.SKILL_ABORT
    assert "moving to target" in result.error
    assert len(env.step_calls) == config.SKILL_STALL_STEPS + 1


def test_drop_run_is_skill_abort() -> None:
    class _DropEnv(FakeEnv):
        def transfer_distances(self, step: dict[str, str]) -> tuple[float, float]:
            if self._steps <= 1:
                return (0.05, 0.8)
            return (config.TRANSFER_DROP_DISTANCE + 0.05, 0.8)

    env = _DropEnv()
    executor = build_executor(env=env, vla=FakeVLA(actions=[[0.0] * 7]))
    result = run_plan(executor, make_plan([_TRANSFER]))
    assert result.reason == StopReason.SKILL_ABORT
    assert "dropped" in result.error
    assert len(env.step_calls) == 2


def test_subtask_timeout_includes_abort_error() -> None:
    env = FakeEnv()
    executor = build_executor(env=env, vla=FakeVLA(actions=[[0.0] * 7]))
    with patch("config.MAX_SKILL_STEPS", {"transfer": 2}):
        result = run_plan(
            executor,
            make_plan(
                [
                    {
                        "skill": "transfer",
                        "object_name": "cube",
                        "target_name": "drawer",
                    }
                ]
            ),
        )
    assert result.reason == StopReason.SUBTASK_TIMEOUT
    assert "exceeded 2 steps" in result.error
