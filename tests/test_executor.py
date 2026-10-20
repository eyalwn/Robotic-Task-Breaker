"""Executor tests with fake env and VLA."""
from __future__ import annotations

from unittest.mock import patch

import pytest

import config
from conftest import FakeEnv, FakeVLA, build_executor, make_plan, run_plan
from executor import EpisodeTimeout, SkillTimeout, StopReason, _validate_skill_step_limits


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
    step_a = {"skill": "open", "object_name": "left drawer"}
    step_b = {"skill": "open", "object_name": "right drawer"}
    env = FakeEnv(
        subtask_after={
            ("open", frozenset(step_a.items())): 1,
            ("open", frozenset(step_b.items())): 2,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([step_a, step_b])
    run_plan(executor, plan, skill_step_limits={"open": 1})
    assert len(vla.calls) == 2
    assert vla.calls[0][2] == "open the left drawer"
    assert vla.calls[1][2] == "open the right drawer"


def test_runs_all_steps_without_early_stop() -> None:
    env = FakeEnv(
        subtask_satisfied={
            _step_key("open", object_name="left drawer"): True,
            _step_key("open", object_name="right drawer"): True,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan(
        [
            {"skill": "open", "object_name": "left drawer"},
            {"skill": "open", "object_name": "right drawer"},
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
    plan = make_plan([{"skill": "open", "object_name": "drawer"}])
    result = run_plan(executor, plan, skill_step_limits={"open": 2})
    assert result.reason == StopReason.SUBTASK_TIMEOUT
    assert result.next_step_idx == 0
    assert len(env.step_calls) == 2


def test_episode_timeout_returns_summary() -> None:
    env = FakeEnv(horizon=2)
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "open", "object_name": "drawer"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.EPISODE_TIMEOUT
    assert result.error == "episode hits maximum steps-budget: 2 steps"
    assert result.next_step_idx == 0
    assert len(env.step_calls) == 2


def test_open_subtask_success_immediate() -> None:
    env = FakeEnv(
        subtask_satisfied={_step_key("open", object_name="right drawer"): True}
    )
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "open", "object_name": "right drawer"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert result.next_step_idx == 1
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_open_calls_vla_and_succeeds() -> None:
    env = FakeEnv(
        subtask_after={_step_key("open", object_name="right drawer"): 2}
    )
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "open", "object_name": "right drawer"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert result.next_step_idx == 1
    assert len(vla.calls) == 1
    assert vla.calls[0][2] == "open the right drawer"
    assert len(env.step_calls) == 2


def test_skill_timeout_contract() -> None:
    """Explicit per-skill caps time out without subtask success."""
    env = FakeEnv()
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "open", "object_name": "drawer"}
    limits = {"open": 3}
    with pytest.raises(SkillTimeout, match="exceeded explicit cap of 3 steps"):
        executor._run_step(step, limits)
    assert len(env.step_calls) == 3


def test_episode_timeout_contract() -> None:
    """Skills without subtask success time out at the episode cap."""
    env = FakeEnv(horizon=2)
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "open", "object_name": "drawer"}
    with pytest.raises(
        EpisodeTimeout, match="episode hits maximum steps-budget: 2 steps"
    ):
        executor._run_step(step, None)
    assert len(env.step_calls) == 2


def test_close_vla_prompt() -> None:
    executor = build_executor(env=FakeEnv(), vla=FakeVLA())
    close = {"skill": "close", "object_name": "left drawer"}
    assert executor._build_vla_task(close) == "close the left drawer"


def test_new_step_at_horizon_does_not_step() -> None:
    first = {"skill": "open", "object_name": "drawer"}
    second = {"skill": "close", "object_name": "drawer"}
    env = FakeEnv(
        horizon=2,
        subtask_after={_step_key("open", object_name="drawer"): 2},
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    result = run_plan(executor, make_plan([first, second]))
    assert result.reason == StopReason.EPISODE_TIMEOUT
    assert result.next_step_idx == 1
    assert len(env.step_calls) == 2
    assert len(vla.calls) == 2


def test_episode_cap_uses_env_horizon() -> None:
    plan = make_plan([{"skill": "open", "object_name": "drawer"}])
    env = FakeEnv(horizon=2)
    executor = build_executor(env=env, vla=FakeVLA(actions=[[0.0] * 7]))
    result = run_plan(executor, plan)
    assert result.reason == StopReason.EPISODE_TIMEOUT
    assert len(env.step_calls) == 2

    env_ok = FakeEnv(
        horizon=2,
        subtask_after={_step_key("open", object_name="drawer"): 1},
    )
    executor_ok = build_executor(env=env_ok, vla=FakeVLA(actions=[[0.0] * 7]))
    result_ok = run_plan(executor_ok, plan)
    assert result_ok.reason == StopReason.PLAN_DONE
    assert len(env_ok.step_calls) == 1


def test_telemetry_threshold_without_hard_stop() -> None:
    env = FakeEnv(horizon=10)
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    with patch("config.MAX_SKILL_STEPS", {"open": 2, "close": 400}):
        result = run_plan(
            executor,
            make_plan([{"skill": "open", "object_name": "drawer"}]),
        )
    assert result.reason == StopReason.EPISODE_TIMEOUT
    assert result.skill_telemetry[0]["threshold_crossed"] is True
    assert result.skill_telemetry[0]["steps"] == 10


def test_skill_done_when_subtask_already_satisfied() -> None:
    env = FakeEnv(
        subtask_satisfied={_step_key("open", object_name="drawer"): True}
    )
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "open", "object_name": "drawer"}
    telemetry = executor._run_step(step, None)
    assert telemetry["threshold_crossed"] is False
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_open_task_string_preserves_object_name_detail() -> None:
    step = {"skill": "open", "object_name": "right drawer"}
    env = FakeEnv(subtask_after={("open", frozenset(step.items())): 1})
    vla = FakeVLA(actions=[[0.0] * 12])
    executor = build_executor(env=env, vla=vla)
    run_plan(executor, [step])
    assert vla.calls[0][2] == "open the right drawer"
    assert len(env.step_calls) == 1


def test_unknown_skill_raises() -> None:
    env = FakeEnv()
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    result = run_plan(
        executor,
        [{"skill": "grab", "object_name": "drawer"}],
    )
    assert result.reason == StopReason.EXECUTION_ERROR


def test_executes_full_returned_chunk() -> None:
    """Executor must not trim actions; it runs every action from the service."""
    chunk_size = 7
    actions = [[float(i)] * 7 for i in range(chunk_size)]
    env = FakeEnv()
    vla = FakeVLA(actions=actions)
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "open", "object_name": "drawer"}
    with pytest.raises(SkillTimeout, match="exceeded explicit cap"):
        executor._run_step(step, {"open": chunk_size})
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
    step = {"skill": "open", "object_name": "drawer"}
    executor._run_step(step, {"open": chunk_size})
    assert len(vla.calls) == 1
    assert len(env.step_calls) == done_after
    assert env.step_calls[0][0] == 0.0
    assert env.step_calls[1][0] == 1.0


def test_env_step_done_reports_plan_done() -> None:
    env = FakeEnvStepDone(done_after=1)
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7, [0.2] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "open", "object_name": "drawer"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(env.step_calls) == 1


def test_subtask_timeout_includes_error() -> None:
    env = FakeEnv()
    executor = build_executor(env=env, vla=FakeVLA(actions=[[0.0] * 7]))
    result = run_plan(
        executor,
        make_plan([{"skill": "open", "object_name": "drawer"}]),
        skill_step_limits={"open": 2},
    )
    assert result.reason == StopReason.SUBTASK_TIMEOUT
    assert "exceeded explicit cap of 2 steps" in result.error


def test_timeout_on_second_step_reports_suffix_index() -> None:
    first = {"skill": "open", "object_name": "left drawer"}
    second = {"skill": "close", "object_name": "left drawer"}
    env = FakeEnv(
        subtask_satisfied={
            _step_key("open", object_name="left drawer"): True,
        }
    )
    executor = build_executor(env=env, vla=FakeVLA(actions=[[0.0] * 7]))
    result = run_plan(
        executor,
        [first, second],
        skill_step_limits={"open": 4, "close": 2},
    )
    assert result.reason == StopReason.SUBTASK_TIMEOUT
    assert result.next_step_idx == 1


def test_validate_skill_step_limits_rejects_unknown_skill() -> None:
    with pytest.raises(ValueError, match="unsupported skill"):
        _validate_skill_step_limits({"grab": 1})


def test_validate_skill_step_limits_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        _validate_skill_step_limits({"open": 0})
