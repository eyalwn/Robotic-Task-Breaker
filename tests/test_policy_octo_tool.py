"""Tests for the Octo VLA HTTP-client policy tool."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sim.maniskill_backend import ManiSkillArm
from vla import VLAConfig, create_vla_manipulation

from conftest import FakeEnv

SERVICE_URL = "http://localhost:8765"
FAKE_ACTION = [0.1, -0.2, 0.3, 0.8]


def _cfg(**kwargs):
    return VLAConfig(**kwargs)


def _make_response(action=None):
    resp = MagicMock()
    resp.json.return_value = {"action": action or FAKE_ACTION}
    resp.raise_for_status.return_value = None
    return resp


class FakeCamera:
    def __init__(self):
        self.calls = 0

    def capture_image(self) -> bytes:
        self.calls += 1
        return b"\xff\xd8\xff\xe0fake_jpeg"


class _EvaluateSuccessEnv(FakeEnv):
    def __init__(self, n_act: int = 4, success_after_steps: int = 2) -> None:
        super().__init__(n_act=n_act)
        self._success_after_steps = success_after_steps

    def evaluate(self):
        # Ignore the ManiSkillArm bootstrap zero-action step when deciding success.
        applied = max(0, len(self.steps) - 1)
        return {"success": 1 if applied >= self._success_after_steps else 0}


@pytest.fixture
def arm_and_camera():
    env = FakeEnv(n_act=4)
    arm = ManiSkillArm(env)
    camera = FakeCamera()
    return arm, camera, env


def test_tool_calls_camera_and_service(arm_and_camera):
    arm, camera, env = arm_and_camera
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=2,
        cfg=_cfg(),
    )

    with patch(
        "vla.client.requests.post",
        return_value=_make_response(),
    ) as mock_post:
        result = tool.invoke({})

    assert camera.calls == 2
    assert mock_post.call_count == 2
    call_kwargs = mock_post.call_args
    assert SERVICE_URL + "/predict" in call_kwargs[0][0]
    assert "image" in call_kwargs[1]["files"]


def test_tool_applies_action_to_arm(arm_and_camera):
    arm, camera, env = arm_and_camera
    scale = 0.05
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=1,
        cfg=_cfg(action_scale=scale),
    )

    with patch("vla.client.requests.post", return_value=_make_response()):
        tool.invoke({})

    # env.steps[0] is the zero-action init step from ManiSkillArm.__init__
    assert len(env.steps) >= 2
    dx_expected = FAKE_ACTION[0] * scale
    actual = env.steps[1]
    assert abs(actual[0] - dx_expected) < 1e-6, f"Expected dx≈{dx_expected}, got {actual[0]}"


def test_tool_opens_gripper_when_positive(arm_and_camera):
    arm, camera, env = arm_and_camera
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=2,
        cfg=_cfg(),
    )

    # Construct a chunk where the gripper first closes strongly, then opens strongly.
    # This exercises the internal gripper_state flip logic (close → open).
    chunk = [
        [0.0, 0.0, 0.0, -2.0],
        [0.0, 0.0, 0.0, 2.0],
    ]
    resp = MagicMock()
    resp.json.return_value = {"actions": chunk}
    resp.raise_for_status.return_value = None

    with patch("vla.client.requests.post", return_value=resp):
        tool.invoke({})

    gripper_actions = [s[3] for s in env.steps]
    assert any(g > 0 for g in gripper_actions), "Expected gripper-open action"


def test_tool_closes_gripper_when_negative(arm_and_camera):
    arm, camera, env = arm_and_camera
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=1,
        cfg=_cfg(),
    )

    # Single strong negative gripper command should close the gripper.
    chunk = [[0.0, 0.0, 0.0, -2.0]]
    resp = MagicMock()
    resp.json.return_value = {"actions": chunk}
    resp.raise_for_status.return_value = None

    with patch("vla.client.requests.post", return_value=resp):
        tool.invoke({})

    gripper_actions = [s[3] for s in env.steps]
    assert any(g < 0 for g in gripper_actions), "Expected gripper-close action"


def test_tool_skips_gripper_when_below_threshold(arm_and_camera):
    arm, camera, env = arm_and_camera
    action_no_gripper = [0.1, 0.1, 0.1, 0.3]  # abs < 0.5 → no gripper call
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=1,
        cfg=_cfg(steps_per_action=1),
    )

    with patch("vla.client.requests.post", return_value=_make_response(action_no_gripper)):
        tool.invoke({})

    # Only move_ee steps expected (gripper dim stays 0)
    gripper_actions = [s[3] for s in env.steps]
    assert all(g == 0 for g in gripper_actions), "Gripper should not have been actuated"


def test_connection_error_returns_string_not_exception(arm_and_camera):
    import requests as req_lib

    arm, camera, env = arm_and_camera
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=5,
        cfg=_cfg(),
    )

    with patch(
        "vla.client.requests.post",
        side_effect=req_lib.exceptions.ConnectionError("refused"),
    ):
        result = tool.invoke({})

    assert isinstance(result, str)
    assert "connection error" in result.lower()


def test_http_error_returns_string_not_exception(arm_and_camera):
    import requests as req_lib

    arm, camera, env = arm_and_camera
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=5,
        cfg=_cfg(),
    )

    bad_resp = MagicMock()
    bad_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError("500")

    with patch("vla.client.requests.post", return_value=bad_resp):
        result = tool.invoke({})

    assert isinstance(result, str)
    assert "error" in result.lower()


def test_returns_structured_reason_code(arm_and_camera):
    arm, camera, _env = arm_and_camera
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=2,
        cfg=_cfg(max_inner_steps=1),
    )

    with patch("vla.client.requests.post", return_value=_make_response()):
        result = tool.invoke({})

    payload = json.loads(result)
    assert payload["reason"] == "max_inner_steps"
    assert payload["steps_applied"] == 1


def test_stops_on_backend_evaluate_success():
    env = _EvaluateSuccessEnv(success_after_steps=2)
    arm = ManiSkillArm(env)
    camera = FakeCamera()
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="Execute a VLA-driven pick sequence by querying the Octo service.",
        task_prompt="pick the cube",
        server_address=SERVICE_URL,
        arm=arm,
        camera=camera,
        max_steps=10,
        cfg=_cfg(steps_per_action=1),
    )

    with patch("vla.client.requests.post", return_value=_make_response()):
        result = tool.invoke({})

    payload = json.loads(result)
    assert payload["status"] == "success"
    assert payload["reason"] == "task_success"
    assert payload["steps_applied"] == 2
