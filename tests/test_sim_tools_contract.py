"""Contract tests for ManiSkillDrive and ManiSkillCamera using a FakeEnv.

No ManiSkill installation required.
"""
from __future__ import annotations
import math
import types
import numpy as np
import pytest

from conftest import FakeEnv


# ---------------------------------------------------------------------------
# Drive tests
# ---------------------------------------------------------------------------

from sim.maniskill_backend import (
	ManiSkillDrive,
	ManiSkillCamera,
	_STEPS_PER_METER,
	_STEPS_PER_DEGREE,
)


def _make_drive():
	env = FakeEnv()
	drive = ManiSkillDrive(env)
	return env, drive


def test_move_forward_step_count():
	distance = 0.5
	env, drive = _make_drive()
	drive.move_linear(distance)
	expected = max(1, int(abs(distance) * _STEPS_PER_METER))
	assert len(env.actions) == expected


def test_move_backward_step_count():
	distance = -0.5
	env, drive = _make_drive()
	drive.move_linear(distance)
	expected = max(1, int(abs(distance) * _STEPS_PER_METER))
	assert len(env.actions) == expected


def test_move_linear_sign():
	env_fwd, drive_fwd = _make_drive()
	drive_fwd.move_linear(0.5)
	env_bwd, drive_bwd = _make_drive()
	drive_bwd.move_linear(-0.5)

	fwd_linear = env_fwd.actions[0][0]
	bwd_linear = env_bwd.actions[0][0]
	assert fwd_linear > 0, "forward should produce positive linear action"
	assert bwd_linear < 0, "backward should produce negative linear action"


def test_turn_in_place_step_count():
	angle = 30.0
	env, drive = _make_drive()
	drive.turn_in_place(angle)
	expected = max(1, int(abs(angle) * _STEPS_PER_DEGREE))
	assert len(env.actions) == expected


def test_turn_left_sign():
	"""Positive angle → positive angular action dim (left/CCW)."""
	env, drive = _make_drive()
	drive.turn_in_place(30.0)
	assert env.actions[0][1] > 0, "positive angle should produce positive angular action"


def test_turn_right_sign():
	"""Negative angle → negative angular action dim (right/CW)."""
	env, drive = _make_drive()
	drive.turn_in_place(-30.0)
	assert env.actions[0][1] < 0, "negative angle should produce negative angular action"


def test_tool_turn_left_dispatches_positive():
	"""tools.create_turn_left must call turn_in_place with a positive value."""
	from sim.tools import create_turn_left

	env, drive = _make_drive()
	tool_fn = create_turn_left(drive)
	tool_fn.invoke({"angle_degrees": 45.0})
	assert env.actions[0][1] > 0


def test_tool_turn_right_dispatches_negative():
	"""tools.create_turn_right must call turn_in_place with a negative value."""
	from sim.tools import create_turn_right

	env, drive = _make_drive()
	tool_fn = create_turn_right(drive)
	tool_fn.invoke({"angle_degrees": 45.0})
	assert env.actions[0][1] < 0


# ---------------------------------------------------------------------------
# Camera test
# ---------------------------------------------------------------------------

def test_capture_image_returns_jpeg():
	import cv2

	env = FakeEnv()
	cam = ManiSkillCamera(env)
	data = cam.capture_image()

	assert isinstance(data, bytes)
	assert len(data) > 0

	arr = np.frombuffer(data, dtype=np.uint8)
	img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	assert img is not None, "JPEG bytes must decode to a valid image"
	assert img.shape[0] > 0 and img.shape[1] > 0
