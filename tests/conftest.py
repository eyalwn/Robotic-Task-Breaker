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
        subtask_satisfied: dict[tuple[str, frozenset], bool] | None = None,
        subtask_after: dict[tuple[str, frozenset], int] | None = None,
        steps: int = 0,
    ) -> None:
        self._steps = steps
        self.step_calls: list[list[float]] = []
        self._subtask_satisfied = subtask_satisfied or {}
        self._subtask_after = subtask_after or {}

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self) -> tuple[bytes, bytes, list[float]]:
        return b"main-image", b"wrist-image", [0.0] * 8

    def step(self, action: list[float]) -> bool:
        self.step_calls.append(list(action))
        self._steps += 1
        return False

    def check_subtask(self, step: dict[str, str]) -> bool:
        skill = step["skill"]
        if skill not in {"grab", "place", "close", "turn_on"}:
            raise KeyError(f"unsupported skill: {skill!r}")
        key = (skill, frozenset(step.items()))
        if key in self._subtask_after:
            return self._steps >= self._subtask_after[key]
        return bool(self._subtask_satisfied.get(key, False))


class FakeVLA:
    """Records predict calls and returns a fixed action chunk."""

    def __init__(self, actions: list[list[float]] | None = None) -> None:
        self.actions = actions or [[0.1] * 7]
        self.calls: list[tuple[bytes, bytes, list[float], str]] = []

    def predict(
        self, image: bytes, wrist: bytes, state: list[float], task: str
    ) -> list[list[float]]:
        self.calls.append((image, wrist, state, task))
        return self.actions


@pytest.fixture
def fake_env() -> FakeEnv:
    return FakeEnv()


def build_executor(
    *,
    env: FakeEnv | None = None,
    vla: FakeVLA | None = None,
) -> Executor:
    executor = Executor(env=env or FakeEnv())  # type: ignore[arg-type]
    executor._vla = vla or FakeVLA()
    return executor


def run_plan(executor: Executor, steps: Plan):
    return executor.run(steps)


def make_plan(steps: list[dict[str, str]] | None = None) -> Plan:
    return list(steps or [])
