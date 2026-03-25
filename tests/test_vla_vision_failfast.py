from __future__ import annotations

import json
from unittest.mock import patch

from conftest import FakeEnv
from sim.maniskill_backend import ManiSkillArm, ManiSkillCamera
from sim.vision_health import VisionHealth
from vla import VLAConfig, create_vla_manipulation


class _DeviceLostEnv:
    action_space = type("_A", (), {"shape": (4,)})()

    def render(self):
        raise RuntimeError("Vulkan ErrorDeviceLost")

    def step(self, _action):
        return {"extra": {"obj_pose": [0.0, 0.0, 0.01]}}, 0.0, False, False, {}

    def reset(self, **_kwargs):
        return None, {}

    @property
    def unwrapped(self):
        return self


def test_camera_marks_session_unusable_after_device_lost():
    env = _DeviceLostEnv()
    camera = ManiSkillCamera(env)

    data = camera.capture_image()

    assert isinstance(data, bytes)
    assert len(data) > 0
    assert camera.last_capture_was_placeholder is True
    assert camera.vision_degraded is True
    assert camera.session_unusable_for_vla is True


class _DegradedCamera:
    vision_health = VisionHealth(
        last_frame_real=False,
        degraded_reason="placeholder_capture",
        session_unusable_for_vla=True,
    )
    session_unusable_for_vla = True
    vision_degraded = True
    last_capture_was_placeholder = True

    def capture_image(self) -> bytes:
        return b"\xff\xd8\xff\xe0fake_jpeg"

    def is_placeholder_jpeg(self, _image_bytes: bytes) -> bool:
        return True


def test_vla_tool_aborts_before_predict_when_vision_unavailable():
    arm = ManiSkillArm(FakeEnv(n_act=4))
    camera = _DegradedCamera()
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="VLA pick",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=5,
        cfg=VLAConfig(fail_on_placeholder=True),
    )

    with patch("vla.client.requests.post") as mock_post:
        result = tool.invoke({})

    payload = json.loads(result)
    assert payload["reason"] == "vision_unavailable"
    assert payload["status"] == "stopped"
    assert mock_post.call_count == 0


def test_vision_health_is_single_source_of_truth():
    health = VisionHealth()
    assert health.vision_unavailable is True
    health.mark_real("obs-dict")
    assert health.vision_unavailable is False
    health.mark_unavailable("vulkan_device_lost")
    assert health.vision_unavailable is True
