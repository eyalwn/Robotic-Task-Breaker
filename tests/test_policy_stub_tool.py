"""Tests for the scripted pick-cube policy stub tool."""
import numpy as np
import pytest

from sim.maniskill_backend import ManiSkillArm
from sim.policy_stub_tools import create_pick_cube_with_policy

from conftest import FakeEnv


def test_policy_stub_calls_ee_and_gripper():
    env = FakeEnv(n_act=4)
    arm = ManiSkillArm(env)
    tool = create_pick_cube_with_policy(arm)

    result = tool.invoke({})

    assert "executed" in result.lower()
    assert len(env.steps) > 5

    ee_used = any(np.any(a[:3] != 0) for a in env.steps)
    gripper_used = any(a[3] != 0 for a in env.steps)
    assert ee_used, "EE dims were never exercised"
    assert gripper_used, "Gripper dim was never exercised"
