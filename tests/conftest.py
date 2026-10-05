"""Shared fixtures for hermetic Longor unit tests."""
from __future__ import annotations

import pytest

from executor import Executor
from plan import Plan


class FakeEnv:
    """Minimal env stub for executor skill tests."""

    def __init__(
        self,
        *,
        subtask_satisfied=None,
        subtask_after=None,
        steps: int = 0,
    ) -> None:
        self._steps = steps
        self.step_calls: list[list[float]] = []
        self._subtask_satisfied = {} if subtask_satisfied is None else subtask_satisfied
        self._subtask_after = {} if subtask_after is None else subtask_after

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self) -> tuple[tuple[bytes, bytes], list[float]]:
        return (b"main-image", b"wrist-image"), [0.0] * 8

    def step(self, action: list[float]) -> bool:
        self.step_calls.append(list(action))
        self._steps += 1
        return False

    def check_subtask(self, step: dict[str, str]) -> bool:
        skill = step["skill"]
        if skill != "open":
            raise KeyError(f"unsupported skill: {skill!r}")
        key = (skill, frozenset(step.items()))
        if key in self._subtask_after:
            return self._steps >= self._subtask_after[key]
        return bool(self._subtask_satisfied.get(key, False))


class FakeVLA:
    """Records predict calls and returns a fixed action chunk."""

    def __init__(self, actions=None) -> None:
        self.actions = actions if actions is not None else [[0.1] * 7]
        self.calls: list[tuple[tuple[bytes, ...], list[float], str]] = []

    def predict(
        self,
        images: tuple[bytes, ...],
        state: list[float],
        task: str,
    ) -> list[list[float]]:
        self.calls.append((images, state, task))
        return self.actions


@pytest.fixture
def fake_env() -> FakeEnv:
    return FakeEnv()


def build_executor(
    *,
    env=None,
    vla=None,
) -> Executor:
    env = env or FakeEnv()
    vla = vla or FakeVLA()
    executor = Executor(env=env)  # type: ignore[arg-type]
    executor._vla = vla
    return executor


def run_plan(executor: Executor, steps: Plan):
    return executor.run(steps)


def make_plan(steps=None) -> Plan:
    return list(steps or [])
