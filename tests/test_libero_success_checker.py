"""Tests for LiberoSuccessChecker."""
import numpy as np

from sim.libero_success_checker import LiberoSuccessChecker


class _Arm:
    def __init__(self, info):
        self.last_info = info


def test_check_false_when_no_info():
    c = LiberoSuccessChecker(_Arm({}))
    assert c.check() is False


def test_check_true_when_is_success():
    c = LiberoSuccessChecker(_Arm({"is_success": True}), consecutive_steps=1)
    assert c.check() is True


def test_check_streak_consecutive():
    arm = _Arm({"is_success": True})
    c = LiberoSuccessChecker(arm, consecutive_steps=2)
    assert c.check() is False
    assert c.check() is True


def test_check_numpy_scalar():
    arm = _Arm({"is_success": np.array([True])})
    c = LiberoSuccessChecker(arm, consecutive_steps=1)
    assert c.check() is True


def test_alternate_success_key():
    arm = _Arm({"success": True})
    c = LiberoSuccessChecker(arm, consecutive_steps=1)
    assert c.check() is True
