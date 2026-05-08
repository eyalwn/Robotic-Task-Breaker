"""Hermetic unit tests for LIBERO sim adapters — no lerobot/libero installed."""
from __future__ import annotations

import builtins

import numpy as np
import pytest

from conftest import FakeEnv
from sim.arm_tools import create_move_ee, create_set_gripper
from sim.libero_backend import LiberoArm, LiberoCamera, create_env
from sim.libero_success_checker import LiberoSuccessChecker


class FakeLiberoEnv(FakeEnv):
    """VecEnv-like stub with ``single_action_space`` and 7-dim OSC actions."""

    def __init__(self) -> None:
        super().__init__(n_act=7)

    @property
    def single_action_space(self):
        return self.action_space


def _flatten_act(entry) -> np.ndarray:
    return np.asarray(entry, dtype=np.float64).reshape(-1)


def test_move_ee_action_format_and_preserves_gripper(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.set_gripper("close", steps=1)
    arm.move_ee(0.1, 0.2, 0.3, steps=1)
    act = _flatten_act(env.step_calls[-1])
    assert act[0] == pytest.approx(0.1)
    assert act[1] == pytest.approx(0.2)
    assert act[2] == pytest.approx(0.3)
    assert np.allclose(act[3:6], 0.0)
    assert act[6] == pytest.approx(-1.0)


def test_set_gripper_open_close_gripper_dim(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.set_gripper("open", steps=1)
    assert _flatten_act(env.step_calls[-1])[6] == pytest.approx(1.0)
    arm.set_gripper("close", steps=1)
    assert _flatten_act(env.step_calls[-1])[6] == pytest.approx(-1.0)


def test_camera_capture_image_jpeg_from_pixels():
    env = FakeLiberoEnv()

    class _Arm:
        last_obs = {"pixels": {"image": np.zeros((8, 8, 3), dtype=np.uint8)}}

    cam = LiberoCamera(env, arm=_Arm())
    data = cam.capture_image()
    assert data.startswith(b"\xff\xd8")
    assert hasattr(cam, "vision_health")


def test_create_env_raises_when_lerobot_unavailable(monkeypatch):
    orig = builtins.__import__

    def _fake(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lerobot" or name.startswith("lerobot."):
            raise ImportError("No module named 'lerobot'")
        return orig(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake)
    with pytest.raises(RuntimeError, match="lerobot"):
        create_env()


def test_arm_tools_smoke_with_libero_arm(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    n_after_init = len(env.step_calls)
    move = create_move_ee(arm)
    grip = create_set_gripper(arm)
    move.invoke({"dx": 0.05, "dy": 0.0, "dz": 0.0})
    grip.invoke({"state": "open"})
    assert len(env.step_calls) > n_after_init
