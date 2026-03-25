"""Hermetic tests for ManiSkillArm – no ManiSkill installation required."""
import numpy as np
import pytest

from conftest import FakeEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arm(shape=(4,)):
    from sim.maniskill_backend import ManiSkillArm
    return ManiSkillArm(FakeEnv(shape))


# ---------------------------------------------------------------------------
# Compatibility guard
# ---------------------------------------------------------------------------

def test_incompatible_action_space_raises():
    from sim.maniskill_backend import ManiSkillArm
    with pytest.raises(RuntimeError, match="expected >=4 dims"):
        ManiSkillArm(FakeEnv(n_act=(3,)))


def test_empty_action_space_raises():
    from sim.maniskill_backend import ManiSkillArm
    with pytest.raises(RuntimeError, match="expected >=4 dims"):
        ManiSkillArm(FakeEnv(n_act=(0,)))


# ---------------------------------------------------------------------------
# move_ee
# ---------------------------------------------------------------------------

def test_move_ee_sets_correct_dims():
    arm = _make_arm()
    arm.move_ee(0.1, 0.0, -0.1)
    assert len(arm._env.steps) >= arm._STEPS_PER_CALL
    recent = arm._env.steps[-arm._STEPS_PER_CALL :]
    for act in recent:
        assert act[0] == pytest.approx(0.1)
        assert act[1] == pytest.approx(0.0)
        assert act[2] == pytest.approx(-0.1)
        # gripper dim must be untouched (zero)
        assert act[3] == pytest.approx(0.0)


def test_move_ee_custom_steps():
    arm = _make_arm()
    arm.move_ee(0.05, 0.05, 0.0, steps=2)
    assert len(arm._env.steps) >= 2
    recent = arm._env.steps[-2:]
    assert len(recent) == 2


def test_move_ee_does_not_mutate_zero_action():
    arm = _make_arm()
    arm.move_ee(1.0, 2.0, 3.0)
    assert np.all(arm._zero_act == 0.0)


# ---------------------------------------------------------------------------
# set_gripper
# ---------------------------------------------------------------------------

def test_set_gripper_open():
    arm = _make_arm()
    arm.set_gripper("open")
    recent = arm._env.steps[-arm._STEPS_PER_CALL :]
    for act in recent:
        assert act[3] == pytest.approx(1.0)
        assert np.all(act[0:3] == 0.0)


def test_set_gripper_close():
    arm = _make_arm()
    arm.set_gripper("close")
    recent = arm._env.steps[-arm._STEPS_PER_CALL :]
    for act in recent:
        assert act[3] == pytest.approx(-1.0)
        assert np.all(act[0:3] == 0.0)


def test_gripper_open_close_opposite_signs():
    arm_o = _make_arm()
    arm_o.set_gripper("open")
    arm_c = _make_arm()
    arm_c.set_gripper("close")
    open_val = arm_o._env.steps[-1][3]
    close_val = arm_c._env.steps[-1][3]
    assert open_val == pytest.approx(-close_val)
    assert open_val > 0  # open is positive by convention


def test_set_gripper_custom_steps():
    arm = _make_arm()
    arm.set_gripper("open", steps=3)
    assert len(arm._env.steps) >= 3
    recent = arm._env.steps[-3:]
    assert len(recent) == 3


def test_set_gripper_invalid_state():
    arm = _make_arm()
    with pytest.raises(ValueError, match="'open' or 'close'"):
        arm.set_gripper("half")


# ---------------------------------------------------------------------------
# step helper
# ---------------------------------------------------------------------------

def test_step_helper_uses_zero_action_by_default():
    arm = _make_arm()
    arm.step(steps=2)
    # Ignore the initial seed step; check the two explicit calls.
    assert len(arm._env.steps) >= 2
    recent = arm._env.steps[-2:]
    for act in recent:
        assert np.all(act == 0.0)


def test_step_helper_with_action_override():
    arm = _make_arm()
    custom = np.array([1.0, 2.0, 3.0, 4.0])
    arm.step(action_override=custom, steps=1)
    # Last step should reflect the custom action (seed step is earlier).
    assert np.allclose(arm._env.steps[-1], custom)


# ---------------------------------------------------------------------------
# Wider action space (>4 dims) still works
# ---------------------------------------------------------------------------

def test_larger_action_space():
    arm = _make_arm(shape=(7,))
    arm.move_ee(0.2, 0.3, 0.4)
    recent = arm._env.steps[-arm._STEPS_PER_CALL :]
    for act in recent:
        assert act[0] == pytest.approx(0.2)
        assert act[3] == pytest.approx(0.0)  # gripper untouched
        assert len(act) == 7
