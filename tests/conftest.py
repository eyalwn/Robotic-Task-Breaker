"""Shared fixtures for hermetic Longor unit tests."""
from __future__ import annotations

import pytest

from executor import Executor, FeedbackDecision, FeedbackOutcome
from plan import Plan
from skills import Skill, SkillRegistry
from vla_client import VLAClient


class FakeEnv:
    """Minimal env stub for executor arm-skill and feedback tests."""

    def __init__(self, *, succeed_after: int | None = 0) -> None:
        self._succeed_after = succeed_after
        self._steps = 0
        self.step_calls: list[list[float]] = []

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


class MockFeedbackEvaluator:
    def __init__(
        self,
        decision: FeedbackDecision,
        reason: str = "",
        assessment: str | None = None,
    ) -> None:
        self.decision = decision
        self.reason = reason
        self.assessment = assessment
        self.last_description: str | None = None

    def evaluate(
        self, image_bytes: bytes, description: str, goal: str
    ) -> FeedbackOutcome:
        self.last_description = description
        return FeedbackOutcome(
            decision=self.decision,
            reason=self.reason,
            assessment=self.assessment,
        )


def mock_image_fn() -> bytes:
    return b"fake-image"


@pytest.fixture
def fake_env() -> FakeEnv:
    return FakeEnv()


@pytest.fixture
def fake_vla() -> FakeVLA:
    return FakeVLA()


def build_executor(
    skill: Skill,
    *,
    env: FakeEnv | None = None,
    vla: FakeVLA | None = None,
    decision: FeedbackDecision = FeedbackDecision.CONTINUE,
    step_budget: int = 30,
    goal: str = "test-goal",
) -> Executor:
    registry = SkillRegistry()
    registry.register(skill)
    return Executor(
        registry=registry,
        feedback_evaluator=MockFeedbackEvaluator(decision),
        image_fn=mock_image_fn,
        goal=goal,
        env=env or FakeEnv(),
        vla=vla or FakeVLA(),
        step_budget=step_budget,
    )


def make_plan(body: dict, goal: str = "g") -> Plan:
    return Plan.model_validate({"goal": goal, "body": body})
