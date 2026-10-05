"""Executor tests with fake env and VLA."""
from __future__ import annotations

from unittest.mock import patch

import pytest

import config
from conftest import FakeEnv, FakeVLA, build_executor, make_plan, run_plan
from executor import EpisodeTimeout, SkillTimeout, StopReason


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
    step_a = {"skill": "open", "target_name": "left drawer"}
    step_b = {"skill": "open", "target_name": "right drawer"}
    env = FakeEnv(
        subtask_after={
            ("open", frozenset(step_a.items())): 1,
            ("open", frozenset(step_b.items())): 2,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([step_a, step_b])
    with patch("config.MAX_SKILL_STEPS", {"open": 1}):
        run_plan(executor, plan)
    assert len(vla.calls) == 2
    assert vla.calls[0][2] == "open the left drawer"
    assert vla.calls[1][2] == "open the right drawer"


def test_runs_all_steps_without_early_stop() -> None:
    env = FakeEnv(
        subtask_satisfied={
            _step_key("open", target_name="left drawer"): True,
            _step_key("open", target_name="right drawer"): True,
        }
    )
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan(
        [
            {"skill": "open", "target_name": "left drawer"},
            {"skill": "open", "target_name": "right drawer"},
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
    plan = make_plan([{"skill": "open", "target_name": "drawer"}])
    with patch("config.MAX_SKILL_STEPS", {"open": 2}):
        result = run_plan(executor, plan)
    assert result.reason == StopReason.SUBTASK_TIMEOUT
    assert len(env.step_calls) == 2


def test_episode_timeout_returns_summary() -> None:
    env = FakeEnv()
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "open", "target_name": "drawer"}])
    with patch("config.MAX_EPISODE_STEPS", 2):
        result = run_plan(executor, plan)
    assert result.reason == StopReason.EPISODE_TIMEOUT
    assert result.error == "episode hits maximum steps-budget: 2 steps"
    assert len(env.step_calls) == 2


def test_open_subtask_success_immediate() -> None:
    env = FakeEnv(
        subtask_satisfied={_step_key("open", target_name="right drawer"): True}
    )
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "open", "target_name": "right drawer"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_open_calls_vla_and_succeeds() -> None:
    env = FakeEnv(
        subtask_after={_step_key("open", target_name="right drawer"): 2}
    )
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "open", "target_name": "right drawer"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(vla.calls) == 1
    assert vla.calls[0][2] == "open the right drawer"
    assert len(env.step_calls) == 2


def test_skill_timeout_contract() -> None:
    """Skills without subtask success time out at the per-skill cap."""
    env = FakeEnv()
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "open", "target_name": "drawer"}
    with patch("config.MAX_SKILL_STEPS", {"open": 3}):
        with pytest.raises(SkillTimeout, match="exceeded 3 steps"):
            executor._run_step(step)
    assert len(env.step_calls) == 3


def test_episode_timeout_contract() -> None:
    """Skills without subtask success time out at the episode cap."""
    env = FakeEnv()
    vla = FakeVLA(actions=[[0.0] * 7])
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "open", "target_name": "drawer"}
    with patch("config.MAX_EPISODE_STEPS", 2):
        with pytest.raises(
            EpisodeTimeout, match="episode hits maximum steps-budget: 2 steps"
        ):
            executor._run_step(step)
    assert len(env.step_calls) == 2


def test_skill_done_when_subtask_already_satisfied() -> None:
    env = FakeEnv(
        subtask_satisfied={_step_key("open", target_name="drawer"): True}
    )
    vla = FakeVLA()
    executor = build_executor(env=env, vla=vla)
    step = {"skill": "open", "target_name": "drawer"}
    executor._run_step(step)
    assert len(vla.calls) == 1
    assert len(env.step_calls) == 1


def test_open_task_string_preserves_target_detail() -> None:
    step = {"skill": "open", "target_name": "right drawer"}
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
    step = {"skill": "open", "target_name": "drawer"}
    with patch("config.MAX_SKILL_STEPS", {"open": chunk_size}):
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
    step = {"skill": "open", "target_name": "drawer"}
    with patch("config.MAX_SKILL_STEPS", {"open": chunk_size}):
        executor._run_step(step)
    assert len(vla.calls) == 1
    assert len(env.step_calls) == done_after
    assert env.step_calls[0][0] == 0.0
    assert env.step_calls[1][0] == 1.0


def test_env_step_done_reports_plan_done() -> None:
    env = FakeEnvStepDone(done_after=1)
    vla = FakeVLA(actions=[[0.0] * 7, [0.1] * 7, [0.2] * 7])
    executor = build_executor(env=env, vla=vla)
    plan = make_plan([{"skill": "open", "target_name": "drawer"}])
    result = run_plan(executor, plan)
    assert result.reason == StopReason.PLAN_DONE
    assert len(env.step_calls) == 1


def test_subtask_timeout_includes_error() -> None:
    env = FakeEnv()
    executor = build_executor(env=env, vla=FakeVLA(actions=[[0.0] * 7]))
    with patch("config.MAX_SKILL_STEPS", {"open": 2}):
        result = run_plan(
            executor,
            make_plan([{"skill": "open", "target_name": "drawer"}]),
        )
    assert result.reason == StopReason.SUBTASK_TIMEOUT
    assert "exceeded 2 steps" in result.error
