"""Hermetic unit tests for LIBERO sim adapters — no lerobot/libero installed."""
from __future__ import annotations

import re

import builtins
import logging
import os

import numpy as np
import pytest

from conftest import FakeEnv
from sim.arm_tools import create_move_ee, create_set_gripper
from sim.libero_backend import (
    LiberoArm,
    LiberoBackend,
    LiberoCamera,
    _agent_state_from_obs,
    _log_episode_horizon,
    _maybe_flip_libero_vla_frame,
    _quat2axisangle,
    create_env,
)
from sim.libero_success_checker import LiberoSuccessChecker


@pytest.fixture(autouse=True)
def _no_libero_settle(monkeypatch):
    monkeypatch.setenv("LIBERO_SETTLE_STEPS", "0")


class FakeLiberoEnv(FakeEnv):
    """VecEnv-like stub with ``single_action_space`` and 7-dim OSC actions."""

    def __init__(self) -> None:
        super().__init__(n_act=7)

    @property
    def single_action_space(self):
        return self.action_space


def _flatten_act(entry) -> np.ndarray:
    return np.asarray(entry, dtype=np.float64).reshape(-1)


def test_apply_vla_action_full_osc_with_rotation(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.apply_vla_action(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.8, steps=1)
    act = _flatten_act(env.step_calls[-1])
    assert act[:6] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert act[6] == pytest.approx(-0.8)
    assert arm.last_ik_debug == {"cause": "osc_direct"}


def test_maybe_flip_libero_vla_frame_when_enabled(monkeypatch):
    monkeypatch.setenv("LIBERO_VLA_IMAGE_FLIP", "1")
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    flipped = _maybe_flip_libero_vla_frame(rgb)
    assert np.array_equal(flipped, rgb[::-1, ::-1])


def test_maybe_flip_libero_vla_frame_disabled(monkeypatch):
    monkeypatch.setenv("LIBERO_VLA_IMAGE_FLIP", "0")
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(_maybe_flip_libero_vla_frame(rgb), rgb)


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
    assert arm.last_ik_debug == {"cause": "osc_direct"}


def test_set_gripper_open_close_gripper_dim(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.set_gripper("open", steps=1)
    assert _flatten_act(env.step_calls[-1])[6] == pytest.approx(1.0)
    arm.set_gripper("close", steps=1)
    assert _flatten_act(env.step_calls[-1])[6] == pytest.approx(-1.0)


def test_prepare_for_grab_opens_gripper_and_steps(monkeypatch):
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "3")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "0")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.set_gripper("close", steps=1)
    arm.last_tcp_pos = [0.0, 0.0, 0.5]
    n_before = len(env.step_calls)
    arm.prepare_for_grab()
    new_calls = env.step_calls[n_before:]
    assert len(new_calls) == 4  # open(1) + descend(3)
    assert _flatten_act(new_calls[0])[6] == pytest.approx(1.0)
    for act in new_calls[1:]:
        flat = _flatten_act(act)
        assert flat[6] == pytest.approx(1.0)
        assert flat[2] < 0.0


def test_prepare_for_grab_low_z_loop(monkeypatch):
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "1")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "5")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_TARGET_Z", "0.55")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_tcp_pos = [0.0, 0.0, 0.8]
    # _eef_z fakes only the low-z loop (start/done logs use _eef_xyz).
    z_reads = [0.8, 0.65, 0.6, 0.5, 0.5]

    def _fake_eef_z():
        return z_reads.pop(0) if z_reads else 0.5

    monkeypatch.setattr(arm, "_eef_z", _fake_eef_z)
    n_before = len(env.step_calls)
    arm.prepare_for_grab()
    new_calls = env.step_calls[n_before:]
    # open(1) + initial descend(1) + low-z loop(3) = 5
    assert len(new_calls) == 5
    for act in new_calls[1:]:
        assert _flatten_act(act)[2] < 0.0


def test_prepare_for_grab_settle_after_descend(monkeypatch):
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "1")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "0")
    monkeypatch.setenv("LIBERO_SETTLE_STEPS", "2")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_tcp_pos = [0.0, 0.0, 0.5]
    n_before = len(env.step_calls)
    arm.prepare_for_grab()
    new_calls = env.step_calls[n_before:]
    assert len(new_calls) == 4  # open(1) + descend(1) + settle(2)
    settle_calls = new_calls[-2:]
    for act in settle_calls:
        flat = _flatten_act(act)
        assert flat[:3] == pytest.approx([0.0, 0.0, 0.0])
        assert flat[6] == pytest.approx(1.0)


def test_prepare_for_grab_xy_retreat_before_descend(monkeypatch):
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "2")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT", "1")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT_STEPS", "2")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT_DELTA", "0.1")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "0")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_tcp_pos = [0.0, 0.0, 0.5]

    def _fake_body_position(_env, name: str):
        if name == "basket_1_main":
            return [0.0, 0.0, 0.5]
        return None

    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        _fake_body_position,
    )
    n_before = len(env.step_calls)
    arm.prepare_for_grab()
    new_calls = env.step_calls[n_before:]
    assert len(new_calls) == 5  # open(1) + retreat(2) + descend(2)
    retreat_calls = new_calls[1:3]
    for act in retreat_calls:
        flat = _flatten_act(act)
        assert flat[0] > 0.0
        assert flat[2] == pytest.approx(0.0)


def test_prepare_for_grab_xy_approach_after_descend(monkeypatch):
    approach_max = 3
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "1")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_MAX_STEPS", str(approach_max))
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_DELTA", "0.1")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_RADIUS", "0.1")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_tcp_pos = [0.0, 0.0, 0.5]

    def _fake_body_position(_env, name: str):
        if name == "cream_cheese_1_main":
            return [0.25, 0.0, 0.5]
        return None

    def _track_tcp(_obs: object) -> None:
        arm.last_tcp_pos = [0.0, 0.0, 0.5]

    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        _fake_body_position,
    )
    monkeypatch.setattr(arm, "_update_positions_from_obs", _track_tcp)
    n_before = len(env.step_calls)
    arm.prepare_for_grab(next_object="cream_cheese_1_main")
    new_calls = env.step_calls[n_before:]
    assert _flatten_act(new_calls[0])[6] == pytest.approx(1.0)
    assert _flatten_act(new_calls[1])[2] < 0.0
    approach_calls = new_calls[2:]
    assert 1 <= len(approach_calls) <= approach_max
    for act in approach_calls:
        flat = _flatten_act(act)
        assert flat[0] > 0.0
        assert flat[2] == pytest.approx(0.0)


def test_prepare_for_grab_logs_horiz_dist_after_approach(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_STEPS", "2")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_DELTA", "0.2")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_RADIUS", "0.15")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_tcp_pos = [0.0, -0.4, 0.5]

    def _fake_body_position(_env, name: str):
        if name == "cream_cheese_1_main":
            return [0.0, 0.0, 0.5]
        return None

    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        _fake_body_position,
    )

    def _preserve_tcp(_obs: object) -> None:
        if arm.last_tcp_pos is None:
            arm.last_tcp_pos = [0.0, -0.4, 0.5]

    monkeypatch.setattr(arm, "_update_positions_from_obs", _preserve_tcp)

    def _track_move(dx: float, dy: float, dz: float, steps: int | None = None) -> None:
        if arm.last_tcp_pos is None:
            arm.last_tcp_pos = [0.0, -0.4, 0.5]
        arm.last_tcp_pos[0] += dx
        arm.last_tcp_pos[1] += dy
        arm.last_tcp_pos[2] += dz

    monkeypatch.setattr(arm, "move_ee", _track_move)

    with caplog.at_level(logging.INFO, logger="sim.libero_backend"):
        arm.prepare_for_grab(next_object="cream_cheese_1_main")

    approach_logs = [
        r.message
        for r in caplog.records
        if "after_xy_approach" in r.message and "horiz_dist=" in r.message
    ]
    assert approach_logs
    horiz_match = re.search(r"horiz_dist=([0-9.]+)", approach_logs[0])
    assert horiz_match is not None
    assert float(horiz_match.group(1)) <= 0.15


def test_approach_xy_bounded_by_max_steps(monkeypatch):
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_MAX_STEPS", "4")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_DELTA", "0.05")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_RADIUS", "0.01")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_tcp_pos = [0.0, 0.0, 0.5]

    def _fake_body_position(_env, name: str):
        if name == "cream_cheese_1_main":
            return [1.0, 0.0, 0.5]
        return None

    def _preserve_tcp(_obs: object) -> None:
        if arm.last_tcp_pos is None:
            arm.last_tcp_pos = [0.0, 0.0, 0.5]

    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        _fake_body_position,
    )
    monkeypatch.setattr(arm, "_update_positions_from_obs", _preserve_tcp)

    approach_moves = 0

    def _track_move(dx: float, dy: float, dz: float, steps: int | None = None) -> None:
        nonlocal approach_moves
        approach_moves += 1
        if arm.last_tcp_pos is None:
            arm.last_tcp_pos = [0.0, 0.0, 0.5]
        arm.last_tcp_pos[0] += dx
        arm.last_tcp_pos[1] += dy

    monkeypatch.setattr(arm, "move_ee", _track_move)
    arm.prepare_for_grab(next_object="cream_cheese_1_main")
    assert approach_moves == 4


def test_prepare_for_grab_descend_before_approach_in_logs(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "1")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_XY_RETREAT", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "0")
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_APPROACH_MAX_STEPS", "1")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_tcp_pos = [0.0, 0.0, 0.5]

    def _fake_body_position(_env, name: str):
        if name == "cream_cheese_1_main":
            return [0.1, 0.0, 0.5]
        return None

    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        _fake_body_position,
    )

    with caplog.at_level(logging.INFO, logger="sim.libero_backend"):
        arm.prepare_for_grab(next_object="cream_cheese_1_main")

    messages = [r.message for r in caplog.records]
    descend_idx = next(
        i for i, m in enumerate(messages) if "after_descend" in m
    )
    approach_idx = next(
        i for i, m in enumerate(messages) if "after_xy_approach" in m
    )
    assert descend_idx < approach_idx


def test_camera_capture_image_jpeg_from_pixels():
    env = FakeLiberoEnv()

    class _Arm:
        last_obs = {"pixels": {"image": np.zeros((8, 8, 3), dtype=np.uint8)}}

    cam = LiberoCamera(env, arm=_Arm())
    data = cam.capture_image()
    assert data.startswith(b"\xff\xd8")
    assert hasattr(cam, "vision_health")


def test_camera_capture_wrist_image_jpeg_from_pixels():
    env = FakeLiberoEnv()

    class _Arm:
        last_obs = {
            "pixels": {
                "image2": np.full((8, 8, 3), 200, dtype=np.uint8),
            }
        }

    cam = LiberoCamera(env, arm=_Arm())
    data = cam.capture_wrist_image()
    assert data.startswith(b"\xff\xd8")


def test_log_episode_horizon_unknown_when_missing(caplog):
    caplog.set_level(logging.DEBUG, logger="sim.libero_backend")

    class _BareEnv:
        pass

    _log_episode_horizon(_BareEnv())
    assert "LIBERO episode horizon: unknown (source=none)" in caplog.text


def test_log_episode_horizon_reads_wrapper_chain(caplog):
    caplog.set_level(logging.DEBUG, logger="sim.libero_backend")

    class _Inner:
        max_episode_steps = 280

    class _Vec:
        envs = [_Inner()]

    _log_episode_horizon(_Vec())
    assert "LIBERO episode horizon: 280" in caplog.text


def test_create_env_raises_when_lerobot_unavailable(monkeypatch):
    orig = builtins.__import__

    def _fake(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lerobot" or name.startswith("lerobot."):
            raise ImportError("No module named 'lerobot'")
        return orig(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake)
    with pytest.raises(RuntimeError, match="lerobot"):
        create_env()


def test_env_step_count_matches_env_steps(monkeypatch):
    monkeypatch.setenv("LIBERO_SETTLE_STEPS", "3")
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    assert arm.env_step_count == len(env.step_calls)
    arm.move_ee(0.1, 0.0, 0.0, steps=2)
    assert arm.env_step_count == len(env.step_calls)


def test_rollout_stops_on_episode_termination(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    step_count = 0

    def step_with_termination(action):
        nonlocal step_count
        step_count += 1
        env.step_calls.append(np.asarray(action, dtype=float))
        obs = {"extra": {"obj_pose": [0.0, 0.0, 0.01]}}
        if step_count == 2:
            return obs, 0.0, np.array([True]), False, {}
        return obs, 0.0, False, False, {}

    env.step = step_with_termination
    arm = LiberoArm(env)
    arm.move_ee(0.1, 0.0, 0.0, steps=10)
    assert len(env.step_calls) == 2
    assert arm.episode_done is True


def test_rollout_noop_when_episode_done(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.episode_done = True
    n_calls = len(env.step_calls)
    arm.move_ee(0.1, 0.2, 0.3, steps=10)
    assert len(env.step_calls) == n_calls


def test_rollout_handles_env_terminated_value_error(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)

    def step_raises_terminated(action):
        raise ValueError("executing action in terminated episode")

    env.step = step_raises_terminated
    arm.move_ee(0.1, 0.0, 0.0, steps=10)
    assert arm.episode_done is True


def test_rollout_reraises_unrelated_value_error(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)

    def step_raises_bad_shape(action):
        env.step_calls.append(np.asarray(action, dtype=float))
        raise ValueError("bad action shape")

    env.step = step_raises_bad_shape
    with pytest.raises(ValueError, match="bad action shape"):
        arm.move_ee(0.1, 0.0, 0.0, steps=1)


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


def test_quat2axisangle_identity():
    axis = _quat2axisangle([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(axis, [0.0, 0.0, 0.0], atol=1e-6)


def test_quat2axisangle_90deg_about_z():
    # 90° rotation about z: quat (0, 0, sin(π/4), cos(π/4))
    import math

    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    axis = _quat2axisangle([0.0, 0.0, s, c])
    np.testing.assert_allclose(axis, [0.0, 0.0, math.pi / 2], atol=1e-5)


def test_agent_state_from_obs_prefers_eef_axis_angle_over_joints_and_agent_pos():
    joints = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)
    gripper_qpos = np.array([0.0123, 0.0456], dtype=np.float32)
    agent_pos = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.99], dtype=np.float32)
    obs = {
        "agent_pos": agent_pos.tolist(),
        "robot_state": {
            "joints": {"pos": joints},
            "eef": {
                "pos": [1.0, 2.0, 3.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
            "gripper": {"qpos": gripper_qpos},
        },
    }
    got = _agent_state_from_obs(obs)
    assert got is not None
    expected = np.array(
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0123, 0.0456],
        dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected)


def test_agent_state_from_obs_uses_agent_pos_when_no_joints():
    expected = np.arange(8, dtype=np.float32)
    obs = {"agent_pos": expected.tolist()}
    got = _agent_state_from_obs(obs)
    assert got is not None
    np.testing.assert_allclose(got, expected)


def test_agent_state_from_obs_falls_back_to_robot_state_eef():
    obs = {
        "robot_state": {
            "eef": {
                "pos": [1.0, 2.0, 3.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
            "gripper": {"qpos": [0.42, 9.9]},
        }
    }
    got = _agent_state_from_obs(obs)
    assert got is not None
    np.testing.assert_allclose(
        got,
        np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.42, 9.9], dtype=np.float32),
    )


def test_agent_state_from_obs_pads_single_gripper_qpos():
    obs = {
        "robot_state": {
            "eef": {
                "pos": [0.0, 0.0, 0.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
            "gripper": {"qpos": [0.5]},
        }
    }
    got = _agent_state_from_obs(obs)
    assert got is not None
    np.testing.assert_allclose(
        got,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32),
    )


def test_agent_state_from_obs_ignores_joints_without_eef():
    joints = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)
    gripper_qpos = np.array([0.0123, 0.0456], dtype=np.float32)
    obs = {
        "robot_state": {
            "joints": {"pos": joints},
            "gripper": {"qpos": gripper_qpos},
        }
    }
    assert _agent_state_from_obs(obs) is None


@pytest.mark.parametrize("obs", [None, {}, []])
def test_agent_state_from_obs_returns_none_for_missing_or_invalid(obs):
    assert _agent_state_from_obs(obs) is None


def test_libero_arm_agent_state_returns_zeros_when_last_obs_none(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_obs = None
    got = arm.agent_state
    np.testing.assert_allclose(got, np.zeros(8, dtype=np.float32))


def test_set_target_object_sets_last_obj_pos(monkeypatch):
    monkeypatch.setenv("LIBERO_EE_DELTA_SCALE", "1.0")
    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        lambda _env, name: [0.1, 0.2, 0.448] if name == "alphabet_soup_1" else None,
    )
    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    assert arm.set_target_object("alphabet_soup_1") is True
    assert arm.last_obj_pos == pytest.approx([0.1, 0.2, 0.448])
    arm._update_positions_from_obs(arm.last_obs)
    assert arm.last_obj_pos == pytest.approx([0.1, 0.2, 0.448])


def test_libero_backend_apply_env_defaults(monkeypatch):
    for key in (
        "VLA_AXIS_SIGNS",
        "VLA_AXIS_REMAP",
        "VLA_USE_ENV_TASK_SUCCESS",
        "VLA_GRIPPER_MODE",
        "VLA_ACTION_SCALE",
        "LIBERO_EE_DELTA_SCALE",
        "VLA_STEPS_PER_ACTION",
        "VLA_ADAPTIVE_SCALE",
        "VLA_CHUNK_STEPS",
        "VLA_STAGNATION_CHUNKS",
        "LIBERO_SETTLE_STEPS",
        "VLA_FAIL_WORSENING_STEPS",
        "VLA_PROXIMITY_GRASP",
    ):
        monkeypatch.delenv(key, raising=False)
    LiberoBackend._apply_env_defaults()
    assert os.environ.get("VLA_USE_ENV_TASK_SUCCESS") == "false"
    assert os.environ.get("VLA_GRIPPER_MODE") == "absolute_open_close"
    assert os.environ.get("VLA_ACTION_SCALE") == "1.0"
    assert os.environ.get("LIBERO_EE_DELTA_SCALE") == "1.0"
    assert os.environ.get("VLA_STEPS_PER_ACTION") == "1"
    assert os.environ.get("VLA_ADAPTIVE_SCALE") == "false"
    assert os.environ.get("VLA_CHUNK_STEPS") == "5"
    assert os.environ.get("VLA_STAGNATION_CHUNKS") == "20"
    assert os.environ.get("LIBERO_SETTLE_STEPS") == "10"
    assert os.environ.get("VLA_FAIL_WORSENING_STEPS") == "200"
    assert os.environ.get("VLA_PROXIMITY_GRASP") == "false"
    assert "VLA_AXIS_SIGNS" not in os.environ
    assert "VLA_AXIS_REMAP" not in os.environ


def test_libero_backend_register_skills(monkeypatch):
    import vla.arm_skills as arm_skills_mod

    from core.skills import SkillRegistry
    from sim.libero_backend import LiberoBackend
    from sim.libero_skills import libero_object_human_name, libero_place_result_hook

    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    camera = LiberoCamera(env, arm=arm)

    backend = LiberoBackend.__new__(LiberoBackend)
    backend._env = env
    backend._arm = arm
    backend._camera = camera
    backend._vla_port = 8765

    recorded: dict = {}
    real_register = arm_skills_mod.register_arm_skills

    def _recording_register_arm_skills(*args, **kwargs):
        recorded["kwargs"] = kwargs
        return real_register(*args, **kwargs)

    monkeypatch.setattr(
        arm_skills_mod,
        "register_arm_skills",
        _recording_register_arm_skills,
    )

    class _StubTool:
        def invoke(self, args):
            return "ok"

    monkeypatch.setattr(
        "vla.create_vla_manipulation",
        lambda **kwargs: _StubTool(),
    )
    monkeypatch.setattr(
        "vla.VLAConfig.from_env",
        lambda: type("Cfg", (), {"max_inner_steps": 10})(),
    )

    registry = SkillRegistry()
    backend.register_skills(registry, "pick the can")

    names = {s.name for s in registry._skills.values()}
    assert "state.list_objects" in names
    assert "perception.locate" in names
    assert "arm.grab" in names
    assert "arm.place" in names
    assert recorded["kwargs"]["arm"] is arm
    assert recorded["kwargs"]["name_mapper"] is libero_object_human_name
    assert recorded["kwargs"]["place_result_hook"] is libero_place_result_hook


def test_libero_backend_success_delegates_to_checker(monkeypatch):
    from sim.libero_backend import LiberoBackend

    env = FakeLiberoEnv()
    arm = LiberoArm(env)
    arm.last_info = {"is_success": True}

    backend = LiberoBackend.__new__(LiberoBackend)
    backend._arm = arm

    checker_calls: list[str] = []

    class _FakeChecker:
        def __init__(self, arm_obj, **kwargs):
            checker_calls.append("init")

        def refresh(self):
            checker_calls.append("refresh")

        def check(self):
            checker_calls.append("check")
            return True

    monkeypatch.setattr(
        "sim.libero_success_checker.LiberoSuccessChecker",
        _FakeChecker,
    )
    assert backend.success() is True
    assert checker_calls == ["init", "refresh", "check"]
