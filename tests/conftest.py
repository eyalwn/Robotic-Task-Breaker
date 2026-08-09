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
        goal_conjuncts: list[list[str]] | None = None,
        conjunct_satisfied: dict[int, bool] | None = None,
        grasping_objects: set[str] | None = None,
        grasp_after: dict[str, int] | None = None,
    ) -> None:
        self._succeed_after = succeed_after
        self._steps = steps
        self.max_steps = max_steps
        self.step_calls: list[list[float]] = []
        self._goal_conjuncts = goal_conjuncts
        self._conjunct_satisfied = conjunct_satisfied
        self._grasping_objects = grasping_objects
        self._grasp_after = grasp_after or {}

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

    def goal_conjuncts(self) -> list[list[str]]:
        return [list(conjunct) for conjunct in (self._goal_conjuncts or [])]

    def eval_goal_conjunct(self, index: int) -> bool:
        if self._conjunct_satisfied is not None:
            return bool(self._conjunct_satisfied.get(index, False))
        return False

    def is_grasping(self, object_name: str) -> bool:
        if self._grasping_objects is not None:
            return object_name in self._grasping_objects
        grasp_after = self._grasp_after.get(object_name)
        if grasp_after is not None:
            return self._steps >= grasp_after
        return False


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
