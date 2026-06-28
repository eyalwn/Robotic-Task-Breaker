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


def test_refresh_calls_arm_step():
    class _Arm:
        def __init__(self) -> None:
            self.last_info = {"is_success": False}
            self.step_calls = 0

        def step(self) -> None:
            self.step_calls += 1
            self.last_info = {"is_success": True}

    arm = _Arm()
    c = LiberoSuccessChecker(arm, consecutive_steps=1)
    assert c.check() is False
    c.refresh()
    assert arm.step_calls == 1
    assert c.check() is True


def test_geometry_gate_rejects_is_success_when_objects_far(monkeypatch):
    monkeypatch.setenv("LIBERO_PLACE_SMOKE_MAX_XY", "0.12")

    positions = {
        "alphabet_soup_1_main": [0.5, 0.0, 0.5],
        "cream_cheese_1_main": [0.5, 0.0, 0.5],
        "basket_1_main": [0.0, 0.0, 0.5],
    }

    class _Env:
        pass

    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        lambda _env, name: positions.get(name),
    )
    c = LiberoSuccessChecker(
        _Arm({"is_success": True}),
        env=_Env(),
        task_objects=("alphabet_soup_1_main", "cream_cheese_1_main"),
    )
    geom_ok, dists = c.task_objects_geometry()
    assert geom_ok is False
    assert c.check() is False
    assert dists["alphabet_soup_1_main"] == 0.5


def test_geometry_gate_passes_when_objects_near_basket(monkeypatch):
    monkeypatch.setenv("LIBERO_PLACE_SMOKE_MAX_XY", "0.12")

    positions = {
        "alphabet_soup_1_main": [0.05, 0.0, 0.5],
        "cream_cheese_1_main": [0.04, 0.0, 0.5],
        "basket_1_main": [0.0, 0.0, 0.5],
    }

    class _Env:
        pass

    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        lambda _env, name: positions.get(name),
    )
    c = LiberoSuccessChecker(
        _Arm({"is_success": True}),
        env=_Env(),
        task_objects=("alphabet_soup_1_main", "cream_cheese_1_main"),
    )
    assert c.task_objects_geometry()[0] is True
    assert c.check() is True


def test_plan_return_indicates_failure():
    from sim.libero_success_checker import plan_return_indicates_failure

    assert plan_return_indicates_failure("Failed to place soup") is True
    assert plan_return_indicates_failure("done") is False
    assert plan_return_indicates_failure(None) is False
