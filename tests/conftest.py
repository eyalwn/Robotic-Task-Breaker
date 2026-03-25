"""Shared test fixtures and stubs for Longor tests."""
from __future__ import annotations

import numpy as np
import pytest


class _FakeActionSpace:
    """Minimal action-space stub.

    *n* may be an ``int`` (action dim count) or a ``tuple`` (shape directly).
    Includes a ``sample()`` so stubs can satisfy callers that call it.
    """

    def __init__(self, n) -> None:
        self.shape = (n,) if isinstance(n, int) else tuple(n)

    def sample(self):
        return np.zeros(self.shape)


class FakeEnv:
    """Minimal env stub compatible with ManiSkillArm, ManiSkillDrive, and
    ManiSkillCamera.

    Recorded calls are stored in ``step_calls``; the properties ``steps`` and
    ``actions`` are aliases so every existing test keeps working regardless of
    which name it uses.
    """

    def __init__(self, n_act: int | tuple = 8) -> None:
        self.action_space = _FakeActionSpace(n_act)
        self.step_calls: list[np.ndarray] = []

    # -- aliases ---------------------------------------------------------

    @property
    def steps(self) -> list[np.ndarray]:
        return self.step_calls

    @property
    def actions(self) -> list[np.ndarray]:
        return self.step_calls

    # -- env interface ---------------------------------------------------

    def step(self, action):
        self.step_calls.append(np.array(action, dtype=float))
        obs = {"extra": {"obj_pose": [0.0, 0.0, 0.01]}}
        return obs, 0.0, False, False, {}

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def reset(self, **kwargs):
        return None, {}

    def get_obs(self):
        return {"extra": {"obj_pose": [0.0, 0.0, 0.01]}}

    @property
    def unwrapped(self):
        return self


@pytest.fixture
def fake_env() -> FakeEnv:
    """Fresh :class:`FakeEnv` instance for each test."""
    return FakeEnv()
