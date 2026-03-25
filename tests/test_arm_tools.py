"""Unit tests for arm_tools and ManiSkillArm – no ManiSkill installation required."""
from __future__ import annotations
import numpy as np
import pytest

from conftest import FakeEnv


# ---------------------------------------------------------------------------
# ManiSkillArm tests
# ---------------------------------------------------------------------------

def test_maniskillarm_rejects_small_action_space():
	from sim.maniskill_backend import ManiSkillArm

	for n in (0, 1, 2, 3):
		env = FakeEnv(n_act=n)
		with pytest.raises(RuntimeError, match="expected >=4 dims"):
			ManiSkillArm(env)


def test_maniskillarm_accepts_4_plus_dims():
	from sim.maniskill_backend import ManiSkillArm

	for n in (4, 7, 8):
		env = FakeEnv(n_act=n)
		arm = ManiSkillArm(env)
		assert arm is not None


# ---------------------------------------------------------------------------
# move_ee tool tests
# ---------------------------------------------------------------------------

def test_move_ee_steps_env_with_nonzero_ee_dims():
	from sim.maniskill_backend import ManiSkillArm
	from sim.arm_tools import create_move_ee

	env = FakeEnv(n_act=8)
	arm = ManiSkillArm(env)
	tool = create_move_ee(arm)

	result = tool.invoke({"dx": 0.1, "dy": 0.0, "dz": 0.05})

	assert isinstance(result, str), "tool must return a string"
	assert len(env.step_calls) >= arm._STEPS_PER_CALL
	recent_calls = env.step_calls[-arm._STEPS_PER_CALL :]
	for act in recent_calls:
		assert act[0] == pytest.approx(0.1), "dim 0 should carry dx"
		assert act[1] == pytest.approx(0.0), "dim 1 should carry dy"
		assert act[2] == pytest.approx(0.05), "dim 2 should carry dz"
		assert act[-1] == pytest.approx(0.0), "gripper dim should be untouched"


def test_move_ee_returns_human_readable_string():
	from sim.maniskill_backend import ManiSkillArm
	from sim.arm_tools import create_move_ee

	env = FakeEnv(n_act=4)
	arm = ManiSkillArm(env)
	tool = create_move_ee(arm)

	result = tool.invoke({"dx": -0.05, "dy": 0.1, "dz": 0.0})
	assert isinstance(result, str)
	assert len(env.step_calls) >= arm._STEPS_PER_CALL


# ---------------------------------------------------------------------------
# set_gripper tool tests
# ---------------------------------------------------------------------------

def test_set_gripper_open_sets_dim3_positive():
	from sim.maniskill_backend import ManiSkillArm
	from sim.arm_tools import create_set_gripper

	env = FakeEnv(n_act=8)
	arm = ManiSkillArm(env)
	tool = create_set_gripper(arm)

	result = tool.invoke({"state": "open"})

	assert isinstance(result, str)
	assert len(env.step_calls) >= arm._STEPS_PER_CALL
	recent_calls = env.step_calls[-arm._STEPS_PER_CALL :]
	for act in recent_calls:
		assert act[3] == pytest.approx(1.0), "dim 3 should be +1.0 when opening"
		assert np.all(act[:3] == 0.0), "EE dims should be zero"
		assert np.all(act[4:] == 0.0), "extra dims should be zero"


def test_set_gripper_close_sets_dim3_negative():
	from sim.maniskill_backend import ManiSkillArm
	from sim.arm_tools import create_set_gripper

	env = FakeEnv(n_act=7)
	arm = ManiSkillArm(env)
	tool = create_set_gripper(arm)

	result = tool.invoke({"state": "close"})

	assert isinstance(result, str)
	assert len(env.step_calls) >= arm._STEPS_PER_CALL
	recent_calls = env.step_calls[-arm._STEPS_PER_CALL :]
	for act in recent_calls:
		assert act[3] == pytest.approx(-1.0), "dim 3 should be -1.0 when closing"
		assert np.all(act[:3] == 0.0), "EE dims should be zero"


def test_set_gripper_returns_human_readable_string():
	from sim.maniskill_backend import ManiSkillArm
	from sim.arm_tools import create_set_gripper

	env = FakeEnv(n_act=4)
	arm = ManiSkillArm(env)
	tool = create_set_gripper(arm)

	for state in ("open", "close"):
		result = tool.invoke({"state": state})
		assert isinstance(result, str)
		assert len(result) > 0


def test_set_gripper_rejects_invalid_state():
	from sim.maniskill_backend import ManiSkillArm
	from sim.arm_tools import create_set_gripper

	env = FakeEnv(n_act=4)
	arm = ManiSkillArm(env)
	tool = create_set_gripper(arm)

	result = tool.invoke({"state": "half"})
	assert "Error" in result
	# ManiSkillArm performs an initial zero-action step in __init__;
	# ensure that calling the tool with an invalid state does not
	# introduce any additional env.step() calls.
	assert len(env.step_calls) == 1
