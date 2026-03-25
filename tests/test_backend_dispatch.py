"""Tests for ManiSkill simulation backend adapters (arm and drive)."""
from __future__ import annotations

import pytest

from conftest import FakeEnv
from sim.maniskill_backend import ManiSkillArm, ManiSkillDrive


def test_maniskill_arm_move_ee_delegates():
    env = FakeEnv(n_act=8)
    arm = ManiSkillArm(env)

    arm.move_ee(0.1, 0.2, 0.3)

    assert len(env.step_calls) >= ManiSkillArm._STEPS_PER_CALL
    recent_calls = env.step_calls[-ManiSkillArm._STEPS_PER_CALL :]
    for act in recent_calls:
        assert act[0] == pytest.approx(0.1)
        assert act[1] == pytest.approx(0.2)
        assert act[2] == pytest.approx(0.3)


def test_maniskill_arm_set_gripper_open_delegates():
    env = FakeEnv(n_act=8)
    arm = ManiSkillArm(env)

    arm.set_gripper("open")

    assert len(env.step_calls) >= ManiSkillArm._STEPS_PER_CALL
    recent_calls = env.step_calls[-ManiSkillArm._STEPS_PER_CALL :]
    for act in recent_calls:
        assert act[3] == pytest.approx(1.0)


def test_maniskill_arm_set_gripper_close_delegates():
    env = FakeEnv(n_act=8)
    arm = ManiSkillArm(env)

    arm.set_gripper("close")

    assert len(env.step_calls) >= ManiSkillArm._STEPS_PER_CALL
    recent_calls = env.step_calls[-ManiSkillArm._STEPS_PER_CALL :]
    for act in recent_calls:
        assert act[3] == pytest.approx(-1.0)


def test_maniskill_drive_move_linear_delegates():
    env = FakeEnv(n_act=8)
    drive = ManiSkillDrive(env)

    drive.move_linear(1.0)

    assert len(env.step_calls) > 1
    for act in env.step_calls[1:]:
        assert act[0] == pytest.approx(1.0)


def test_maniskill_drive_turn_in_place_delegates():
    env = FakeEnv(n_act=8)
    drive = ManiSkillDrive(env)

    drive.turn_in_place(90.0)

    assert len(env.step_calls) > 1
    for act in env.step_calls[1:]:
        assert act[1] == pytest.approx(1.0)

