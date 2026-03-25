"""Hermetic tests for PickCubeSuccessChecker (no ManiSkill needed)."""
from __future__ import annotations

import logging

import pytest

from sim.success_checker import PickCubeSuccessChecker


class FakeEnv:
    """Env stub whose cube height is controllable from the test."""

    def __init__(self, height: float = 0.0) -> None:
        self.height = height

    def get_obs(self):
        return {"extra": {"obj_pose": [0.0, 0.0, self.height]}}


class BrokenEnv:
    """Env whose get_obs always raises."""

    def get_obs(self):
        raise RuntimeError("sensor offline")


def test_below_threshold_returns_false():
    env = FakeEnv(height=0.01)
    checker = PickCubeSuccessChecker(env, height_threshold=0.04,
                                     consecutive_steps=3)
    for _ in range(10):
        assert checker.check() is False


def test_requires_consecutive_steps():
    env = FakeEnv(height=0.10)
    checker = PickCubeSuccessChecker(env, height_threshold=0.04,
                                     consecutive_steps=5)
    for _ in range(4):
        assert checker.check() is False
    assert checker.check() is True


def test_streak_resets_on_drop():
    env = FakeEnv(height=0.10)
    checker = PickCubeSuccessChecker(env, height_threshold=0.04,
                                     consecutive_steps=3)
    checker.check()
    checker.check()
    env.height = 0.01
    checker.check()
    env.height = 0.10
    for _ in range(2):
        assert checker.check() is False
    assert checker.check() is True


def test_broken_env_returns_false_and_warns(caplog):
    env = BrokenEnv()
    checker = PickCubeSuccessChecker(env, consecutive_steps=1)
    with caplog.at_level(logging.WARNING):
        assert checker.check() is False
    assert "Cannot determine cube pose" in caplog.text


def test_warning_logged_only_once(caplog):
    env = BrokenEnv()
    checker = PickCubeSuccessChecker(env, consecutive_steps=1)
    with caplog.at_level(logging.WARNING):
        checker.check()
        checker.check()
        checker.check()
    assert caplog.text.count("Cannot determine cube pose") == 1
