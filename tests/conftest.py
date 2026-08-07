"""Shared fixtures for hermetic Longor unit tests."""
from __future__ import annotations

import pytest

from executor import Executor
from plan import Plan
from skills import Skill, SkillRegistry


class FakeEnv:
    """Minimal env stub for executor arm-skill and feedback tests."""

    def __init__(
        self,
        *,
        succeed_after: int | None = 0,
        max_steps: int = 520,
        steps: int = 0,
    ) -> None:
        self._succeed_after = succeed_after
        self._steps = steps
        self.max_steps = max_steps
        self.step_calls: list[list[float]] = []

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self) -> tuple[bytes, bytes, list[float]]:
        return b"main-image", b"wrist-image", [0.0] * 8

    def step(self, action: list[float]) -> bool:
        self.step_calls.append(list(action))
        self._steps += 1
        return True

    def success(self) -> bool:
        if self._succeed_after is None:
            return False
        return self._steps >= self._succeed_after


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


class MockSkill(Skill):
    name = "mock.skill"
    description = "mock skill for tests"
    parameters = []
    returns = "Any"

    def __init__(self, return_value: object = None) -> None:
        self.return_value = return_value
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.return_value


@pytest.fixture
def fake_env() -> FakeEnv:
    return FakeEnv()


def build_executor(
    skill: Skill,
    *,
    env: FakeEnv | None = None,
    vla: FakeVLA | None = None,
) -> Executor:
    registry = SkillRegistry()
    registry.register(skill)
    executor = Executor(registry=registry, env=env or FakeEnv())  # type: ignore[arg-type]
    executor._vla = vla or FakeVLA()
    return executor


def run_plan(executor: Executor, plan: Plan):
    return executor.run(plan)


def make_plan(body: dict, goal: str = "g") -> Plan:
    return Plan.model_validate({"goal": goal, "body": body})
