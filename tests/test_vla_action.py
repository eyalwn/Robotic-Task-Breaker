from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock, patch

from vla import VLAConfig, create_vla_manipulation
from vla.action import process_raw_action, should_flip_gripper
from vla.tools import _axis_calibration_hint


class _FakeCamera:
    def capture_image(self) -> bytes:
        return b"\xff\xd8\xff\xe0fake_jpeg"


class _FakeArm:
    def __init__(self, eef_xyz: list[float], obj_xyz: list[float]) -> None:
        self.last_tcp_pos = list(eef_xyz)
        self.last_obj_pos = list(obj_xyz)
        self.last_obs = None
        self.moves: list[tuple[float, float, float]] = []

    def move_ee(self, dx: float, dy: float, dz: float, steps: int = 1) -> None:
        self.moves.append((dx, dy, dz))
        self.last_tcp_pos[0] += dx
        self.last_tcp_pos[1] += dy
        self.last_tcp_pos[2] += dz

    def set_gripper(self, state: str, steps: int = 1) -> None:
        return None


def _make_response(actions):
    resp = MagicMock()
    resp.json.return_value = {"actions": actions}
    resp.raise_for_status.return_value = None
    return resp


def test_process_raw_action_order_clip_remap_signs_scale():
    cfg = VLAConfig(
        action_scale=0.1,
        action_clip=1.0,
        axis_remap=[2, 0, 1],
        axis_signs=[-1, 1, -1],
    )
    dx, dy, dz = process_raw_action([2.0, -3.0, 0.5], cfg)

    assert (dx, dy, dz) == (-0.05, 0.1, 0.1)


def test_auto_calibrate_applies_when_unset_and_above_threshold():
    arm = _FakeArm(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[0.1, 0.0, 0.0])
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=0.05,
        axis_remap=None,
        axis_signs=None,
        auto_calibrate=True,
        auto_calibrate_min_cosine=0.7,
        grasp_dist=0.0,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=1,
        cfg=cfg,
    )

    # Raw +y should be remapped/flipped to +x with [0,2,1] and [-1,-1,-1].
    with patch(
        "vla.client.requests.post",
        return_value=_make_response([[0.0, -1.0, 0.0, 0.0]]),
    ):
        tool.invoke({})

    assert len(arm.moves) == 1
    dx, dy, dz = arm.moves[0]
    assert dx > 0.0
    assert abs(dy) < 1e-12
    assert abs(dz) < 1e-12


def test_auto_calibrate_overrides_preset_axes_when_enabled():
    """VLA_AUTO_CALIBRATE must apply hint even when VLA_AXIS_* are set in config."""
    arm = _FakeArm(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[0.1, 0.0, 0.0])
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=0.05,
        axis_remap=[0, 1, 2],
        axis_signs=[1.0, 1.0, 1.0],
        auto_calibrate=True,
        auto_calibrate_min_cosine=0.7,
        grasp_dist=0.0,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=1,
        cfg=cfg,
    )

    with patch(
        "vla.client.requests.post",
        return_value=_make_response([[0.0, -1.0, 0.0, 0.0]]),
    ):
        tool.invoke({})

    assert len(arm.moves) == 1
    dx, dy, dz = arm.moves[0]
    assert dx > 0.0
    assert abs(dy) < 1e-12
    assert abs(dz) < 1e-12


def test_axis_hint_log_permutation_improves_cosine_alignment():
    # Synthetic case mirroring the run log where hint suggested:
    # best_remap=[2,0,1], best_signs=[1,-1,-1]
    eef = [0.0, 0.0, 0.0]
    obj = [0.8, 0.3, -0.5]
    # Construct raw so applying [2,0,1] + [1,-1,-1] recovers ideal_dir exactly.
    raw_xyz = [-0.3, 0.5, 0.8]

    best_cos, best_perm, best_signs = _axis_calibration_hint(eef, obj, raw_xyz)
    assert list(best_perm) == [2, 0, 1]
    assert list(best_signs) == [1, -1, -1]

    # Current (unmapped) cosine should be much worse than hinted alignment.
    raw_norm = (raw_xyz[0] ** 2 + raw_xyz[1] ** 2 + raw_xyz[2] ** 2) ** 0.5
    ideal_norm = (obj[0] ** 2 + obj[1] ** 2 + obj[2] ** 2) ** 0.5
    current_cos = sum(a * b for a, b in zip(raw_xyz, obj)) / (raw_norm * ideal_norm)
    assert best_cos > current_cos
    assert best_cos > 0.99


def test_stagnation_not_triggered_when_min_dist_improves_across_chunks():
    arm = _FakeArm(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[1.0, 0.0, 0.0])
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        adaptive_scale=False,
        grasp_dist=0.0,
        stagnation_chunks=2,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=6,
        cfg=cfg,
    )

    # Each response is one chunk. Chunk-end distance is flat, but the best
    # in-chunk distance improves across chunks.
    responses = [
        _make_response([[0.20, 0.0, 0.0, 0.0], [-0.19, 0.0, 0.0, 0.0]]),
        _make_response([[0.21, 0.0, 0.0, 0.0], [-0.21, 0.0, 0.0, 0.0]]),
        _make_response([[0.22, 0.0, 0.0, 0.0], [-0.22, 0.0, 0.0, 0.0]]),
    ]
    with patch("vla.client.requests.post", side_effect=responses):
        result = tool.invoke({})

    payload = json.loads(result)
    assert payload["reason"] == "max_inner_steps"
    assert payload["steps_applied"] == 6


def test_should_flip_gripper_uses_stricter_gate_when_confidence_above_threshold():
    cfg = VLAConfig(gripper_threshold=0.5, gripper_confidence=1.5)

    # State differs ("open" request vs current "close"), but confidence gate blocks it.
    assert not should_flip_gripper(raw_gripper=1.2, current_state="close", cfg=cfg)

    # Equal to confidence still blocked because flip requires strictly greater magnitude.
    assert not should_flip_gripper(raw_gripper=1.5, current_state="close", cfg=cfg)

    # Above max(threshold, confidence) allows the flip.
    assert should_flip_gripper(raw_gripper=1.6, current_state="close", cfg=cfg)


def test_should_flip_gripper_uses_threshold_when_higher_than_confidence():
    cfg = VLAConfig(gripper_threshold=1.2, gripper_confidence=0.5)

    assert not should_flip_gripper(raw_gripper=1.19, current_state="close", cfg=cfg)
    assert should_flip_gripper(raw_gripper=1.21, current_state="close", cfg=cfg)


def test_should_flip_gripper_handles_oversized_raw_input_safely():
    cfg = VLAConfig(action_clip=2.0, gripper_threshold=0.5, gripper_confidence=1.0)

    # Oversized positive value should still map to "open" and be treated as high confidence.
    assert should_flip_gripper(raw_gripper=4.236, current_state="close", cfg=cfg)

    # Same oversized signal should not trigger a flip if state already matches.
    assert not should_flip_gripper(raw_gripper=4.236, current_state="open", cfg=cfg)


def test_vla_gripper_confidence_env_above_range_clamps_to_default(caplog):
    caplog.set_level(logging.WARNING)
    original = os.environ.get("VLA_GRIPPER_CONFIDENCE")
    os.environ["VLA_GRIPPER_CONFIDENCE"] = "10.0"
    try:
        cfg = VLAConfig.from_env()
    finally:
        if original is None:
            os.environ.pop("VLA_GRIPPER_CONFIDENCE", None)
        else:
            os.environ["VLA_GRIPPER_CONFIDENCE"] = original

    assert cfg.gripper_confidence == VLAConfig.gripper_confidence
    assert "out of expected range" in caplog.text

