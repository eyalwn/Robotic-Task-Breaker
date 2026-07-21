from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from vla import VLAConfig, create_vla_manipulation
from vla.action import process_raw_action, requested_gripper_state, should_flip_gripper
from vla.tools import _axis_calibration_hint, _use_raw_openpi_action_path


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


def test_unset_axis_signs_enables_raw_openpi_action_path():
    for env_value in (None, ""):
        original_signs = os.environ.get("VLA_AXIS_SIGNS")
        original_remap = os.environ.get("VLA_AXIS_REMAP")
        try:
            if env_value is None:
                os.environ.pop("VLA_AXIS_SIGNS", None)
            else:
                os.environ["VLA_AXIS_SIGNS"] = env_value
            os.environ.pop("VLA_AXIS_REMAP", None)
            os.environ["VLA_ADAPTIVE_SCALE"] = "false"
            cfg = VLAConfig.from_env()
        finally:
            if original_signs is None:
                os.environ.pop("VLA_AXIS_SIGNS", None)
            else:
                os.environ["VLA_AXIS_SIGNS"] = original_signs
            if original_remap is None:
                os.environ.pop("VLA_AXIS_REMAP", None)
            else:
                os.environ["VLA_AXIS_REMAP"] = original_remap
            os.environ.pop("VLA_ADAPTIVE_SCALE", None)

        assert cfg.axis_signs is None
        assert cfg.axis_remap is None
        assert _use_raw_openpi_action_path(cfg)


def test_process_raw_action_libero_axis_signs_flip_positive_dz():
    """LIBERO pi05_libero: VLA_AXIS_SIGNS=1,1,-1 negates positive raw dz."""
    cfg = VLAConfig(
        action_scale=1.0,
        action_clip=2.0,
        axis_remap=None,
        axis_signs=[1, 1, -1],
    )
    dx, dy, dz = process_raw_action([0.0, 0.0, 0.9], cfg)
    assert (dx, dy, dz) == (0.0, 0.0, -0.9)


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


def test_requested_gripper_state_absolute_denorms_minus_one_to_open():
    cfg = VLAConfig(gripper_mode="absolute_open_close", gripper_threshold=0.5)
    assert requested_gripper_state(-1.0, cfg) == "open"


def test_requested_gripper_state_absolute_denorms_plus_one_to_close():
    cfg = VLAConfig(gripper_mode="absolute_open_close", gripper_threshold=0.5)
    assert requested_gripper_state(1.0, cfg) == "close"


def test_requested_gripper_state_absolute_denorms_high_positive_to_close():
    cfg = VLAConfig(gripper_mode="absolute_open_close", gripper_threshold=0.5)
    assert requested_gripper_state(0.99, cfg) == "close"


def test_should_flip_gripper_absolute_mode_closes_on_high_raw():
    cfg = VLAConfig(
        gripper_mode="absolute_open_close",
        gripper_threshold=0.5,
        gripper_confidence=0.5,
    )
    assert should_flip_gripper(raw_gripper=0.99, current_state="open", cfg=cfg)
    assert not should_flip_gripper(raw_gripper=0.99, current_state="close", cfg=cfg)
    assert should_flip_gripper(raw_gripper=-0.8, current_state="close", cfg=cfg)


def test_maybe_apply_gripper_absolute_mode_applies_close():
    from vla.tools import _ChunkState, _RunState, _maybe_apply_gripper

    class _GripperArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.5], obj_xyz=[0.0, 0.0, 0.45])
            self.gripper_calls: list[str] = []

        def set_gripper(self, state: str, steps: int = 1) -> None:
            self.gripper_calls.append(state)

    cfg = VLAConfig(
        gripper_mode="absolute_open_close",
        gripper_threshold=0.5,
        sim_pickcube_gripper_profile=False,
        steps_per_action=1,
    )
    arm = _GripperArm()
    state = _RunState(gripper_state="open")
    _maybe_apply_gripper(arm, cfg, state, _ChunkState(), raw_gripper=0.99, dist=0.05)
    assert arm.gripper_calls == ["close"]
    assert state.gripper_state == "close"


def test_tool_impl_uses_7dim_apply_policy_action_with_rotation():
    class _LiberoArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[0.1, 0.0, 0.0])
            self.policy_calls: list[tuple] = []

        def apply_policy_action(self, action, steps=1) -> None:
            self.policy_calls.append((list(action), steps))
            self.move_ee(action[0], action[1], action[2], steps=steps)

    arm = _LiberoArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        axis_remap=None,
        axis_signs=None,
        adaptive_scale=False,
        grasp_dist=0.0,
        max_inner_steps=1,
        steps_per_action=1,
        sim_pickcube_gripper_profile=False,
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
        return_value=_make_response([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7]]),
    ):
        tool.invoke({})

    assert len(arm.policy_calls) == 1
    action, steps = arm.policy_calls[0]
    assert action == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7])
    assert steps == 1


def test_tool_impl_falls_back_to_apply_vla_action_without_policy_method():
    class _LiberoArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[0.1, 0.0, 0.0])
            self.vla_calls: list[tuple] = []

        def apply_vla_action(
            self, dx, dy, dz, drx, dry, drz, grip, steps=1
        ) -> None:
            self.vla_calls.append((dx, dy, dz, drx, dry, drz, grip, steps))
            self.move_ee(dx, dy, dz, steps=steps)

    arm = _LiberoArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        axis_remap=None,
        axis_signs=None,
        adaptive_scale=False,
        grasp_dist=0.0,
        max_inner_steps=1,
        sim_pickcube_gripper_profile=False,
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
        return_value=_make_response([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7]]),
    ):
        tool.invoke({})

    assert len(arm.vla_calls) == 1
    dx, dy, dz, drx, dry, drz, grip, steps = arm.vla_calls[0]
    assert (dx, dy, dz) == pytest.approx((0.1, 0.2, 0.3))
    assert (drx, dry, drz) == pytest.approx((0.4, 0.5, 0.6))
    assert grip == pytest.approx(-0.7)


def test_chunk_steps_limits_actions_before_repredict():
    class _LiberoArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[0.1, 0.0, 0.0])
            self.vla_calls: list[tuple] = []

        def apply_vla_action(
            self, dx, dy, dz, drx, dry, drz, grip, steps=1
        ) -> None:
            self.vla_calls.append((dx, dy, dz, drx, dry, drz, grip, steps))
            self.move_ee(dx, dy, dz, steps=steps)

    arm = _LiberoArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        action_clip=20.0,
        axis_remap=None,
        axis_signs=None,
        adaptive_scale=False,
        chunk_steps=5,
        grasp_dist=0.0,
        max_inner_steps=6,
        sim_pickcube_gripper_profile=False,
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

    chunk_16 = [[float(i)] * 7 for i in range(16)]
    predict_calls = 0

    class _CountingClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def predict(self, _image, _task, **_kwargs):
            nonlocal predict_calls
            predict_calls += 1
            return chunk_16

    with patch("vla.tools.VLAClient", _CountingClient):
        tool.invoke({})

    assert len(arm.vla_calls) == 6
    assert predict_calls == 2
    assert arm.vla_calls[0][0] == pytest.approx(0.0)
    assert arm.vla_calls[4][0] == pytest.approx(4.0)
    assert arm.vla_calls[5][0] == pytest.approx(0.0)


def test_tool_impl_uses_task_override():
    arm = _FakeArm(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[0.1, 0.0, 0.0])
    camera = _FakeCamera()
    cfg = VLAConfig(max_inner_steps=1, grasp_dist=0.0, sim_pickcube_gripper_profile=False)
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="default goal",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=1,
        cfg=cfg,
    )
    tasks: list[str] = []

    class _FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def predict(self, _image, task, **_kwargs):
            tasks.append(task)
            return [[0.0, 0.0, 0.0, 0.0]]

    with patch("vla.tools.VLAClient", _FakeClient):
        tool.invoke({"task": "pick up the soup can"})

    assert tasks == ["pick up the soup can"]


def test_task_success_ignored_when_env_flag_false():
    """LIBERO full-task is_success must not end a per-skill grab early."""
    class _EnvSuccessArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.5], obj_xyz=[0.5, 0.0, 0.45])
            self.last_info = {"is_success": True}
            self.vla_calls: list[tuple] = []

        def apply_vla_action(
            self, dx, dy, dz, drx, dry, drz, grip, steps=1
        ) -> None:
            self.vla_calls.append((dx, dy, dz, drx, dry, drz, grip, steps))
            self.move_ee(dx, dy, dz, steps=steps)

    arm = _EnvSuccessArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        axis_remap=None,
        axis_signs=None,
        adaptive_scale=False,
        grasp_dist=0.0,
        max_inner_steps=3,
        use_env_task_success=False,
        sim_pickcube_gripper_profile=False,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=3,
        cfg=cfg,
    )

    with patch(
        "vla.client.requests.post",
        return_value=_make_response([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    ):
        result = tool.invoke({})

    payload = json.loads(result)
    assert payload["reason"] != "task_success"
    assert payload["steps_applied"] == 3
    assert payload["min_dist"] is not None
    assert payload["min_dist"] > 0.4  # arm never moved toward object


def test_proximity_without_lift_returns_not_confirmed():
    from vla.tools import _maybe_proximity_grasp, _RunState

    class _ProxArm(_FakeArm):
        def step(self, steps: int = 1) -> None:
            return None

    arm = _ProxArm(eef_xyz=[0.0, 0.0, 0.5], obj_xyz=[0.0, 0.0, 0.48])
    cfg = VLAConfig(grasp_dist=0.05)
    state = _RunState(steps_applied=10, min_dist=0.02)
    result = _maybe_proximity_grasp(arm, cfg, state, dist=0.02, obj=[0.0, 0.0, 0.48])
    assert result is not None
    detail, lifted = result
    assert not lifted
    assert "grasp not confirmed" in detail


def test_proximity_grasp_disabled_on_openpi_path():
    from vla.tools import _proximity_grasp_enabled

    cfg = VLAConfig(
        proximity_grasp=True,
        adaptive_scale=False,
        axis_remap=None,
        axis_signs=None,
    )
    assert not _proximity_grasp_enabled(cfg)


def test_proximity_grasp_enabled_for_maniskill_profile():
    from vla.tools import _proximity_grasp_enabled

    cfg = VLAConfig(
        proximity_grasp=True,
        adaptive_scale=True,
        axis_remap=None,
        axis_signs=None,
    )
    assert _proximity_grasp_enabled(cfg)


def test_proximity_grasp_disabled_when_env_false():
    from vla.tools import _proximity_grasp_enabled

    cfg = VLAConfig(proximity_grasp=False, adaptive_scale=True)
    assert not _proximity_grasp_enabled(cfg)


def test_openpi_loop_skips_proximity_when_close():
    class _CloseArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.5], obj_xyz=[0.0, 0.0, 0.49])
            self.vla_calls: list[tuple] = []

        def apply_vla_action(
            self, dx, dy, dz, drx, dry, drz, grip, steps=1
        ) -> None:
            self.vla_calls.append((dx, dy, dz, drx, dry, drz, grip, steps))

    arm = _CloseArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        axis_remap=None,
        axis_signs=None,
        adaptive_scale=False,
        proximity_grasp=False,
        grasp_dist=0.05,
        max_inner_steps=2,
        use_env_task_success=False,
        sim_pickcube_gripper_profile=False,
        chunk_steps=1,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=2,
        cfg=cfg,
    )

    with patch(
        "vla.client.requests.post",
        return_value=_make_response([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]),
    ):
        result = tool.invoke({})

    payload = json.loads(result)
    assert payload["reason"] != "proximity_no_lift"
    assert payload["reason"] != "proximity_grasp"
    assert payload["steps_applied"] == 2
    assert len(arm.vla_calls) == 2


def test_lift_delta_returns_grasp_lift_with_confirmed():
    class _LiftArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.5], obj_xyz=[0.1, 0.0, 0.45])
            self._obj_z = 0.45
            self.vla_calls: list[tuple] = []

        def apply_vla_action(
            self, dx, dy, dz, drx, dry, drz, grip, steps=1
        ) -> None:
            self.vla_calls.append((dx, dy, dz, drx, dry, drz, grip, steps))
            self._obj_z += 0.03
            self.last_obj_pos = [0.1, 0.0, self._obj_z]

    arm = _LiftArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        axis_remap=None,
        axis_signs=None,
        adaptive_scale=False,
        proximity_grasp=False,
        success_lift_delta=0.02,
        max_inner_steps=3,
        use_env_task_success=False,
        sim_pickcube_gripper_profile=False,
        chunk_steps=1,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=3,
        cfg=cfg,
    )

    with patch(
        "vla.client.requests.post",
        return_value=_make_response([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]),
    ):
        result = tool.invoke({})

    payload = json.loads(result)
    assert payload["reason"] == "grasp_lift"
    assert payload["grasp_confirmed"] is True
    assert payload["status"] == "success"


def test_maniskill_proximity_still_triggers_when_enabled():
    class _ProxArm(_FakeArm):
        gripper_calls: list[str] = []

        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.5], obj_xyz=[0.0, 0.0, 0.48])
            self._obj_z = 0.48

        def set_gripper(self, state: str, steps: int = 1) -> None:
            self.gripper_calls.append(state)

        def step(self, steps: int = 1) -> None:
            self._obj_z += 0.01
            self.last_obj_pos = [0.0, 0.0, self._obj_z]

    arm = _ProxArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=0.05,
        axis_remap=[0, 1, 2],
        adaptive_scale=True,
        proximity_grasp=True,
        grasp_dist=0.05,
        max_inner_steps=1,
        use_env_task_success=False,
        sim_pickcube_gripper_profile=False,
        chunk_steps=1,
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
        return_value=_make_response([[0.0, 0.0, 0.0, 0.0]]),
    ):
        result = tool.invoke({})

    payload = json.loads(result)
    assert payload["reason"] == "proximity_grasp"
    assert payload["grasp_confirmed"] is True


def test_adaptive_scaling_factor_decreases_with_dist():
    arm = _FakeArm(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[0.12, 0.0, 0.0])
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=0.05,
        adaptive_scale=True,
        adaptive_ref_dist=0.15,
        adaptive_min_factor=0.2,
        grasp_dist=0.0,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=3,
        cfg=cfg,
    )

    actions = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    with patch("vla.client.requests.post", return_value=_make_response(actions)):
        tool.invoke({})

    dxs = [dx for dx, _, _ in arm.moves]
    assert len(dxs) == 3
    assert dxs[0] > dxs[1] > dxs[2]
    min_allowed = cfg.action_scale * cfg.adaptive_min_factor
    assert all(dx >= min_allowed - 1e-12 for dx in dxs)


def test_adaptive_scaling_can_be_disabled():
    arm = _FakeArm(eef_xyz=[0.0, 0.0, 0.0], obj_xyz=[0.12, 0.0, 0.0])
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=0.05,
        adaptive_scale=False,
        adaptive_ref_dist=0.15,
        adaptive_min_factor=0.2,
        grasp_dist=0.0,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=3,
        cfg=cfg,
    )

    actions = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    with patch("vla.client.requests.post", return_value=_make_response(actions)):
        tool.invoke({})

    dxs = [dx for dx, _, _ in arm.moves]
    assert len(dxs) == 3
    assert dxs[0] == dxs[1] == dxs[2] == cfg.action_scale


def test_vla_debug_action_telemetry_logs_first_steps(monkeypatch, caplog):
    class _LiberoArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.5], obj_xyz=[0.1, 0.0, 0.448])
            self._pos_scale = 1.0
            self._gripper_value = 1.0
            self.last_osc_action = None
            self.policy_calls: list[tuple] = []

        @property
        def agent_state(self):
            import numpy as np

            return np.array(
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, -1.0],
                dtype=np.float32,
            )

        def apply_policy_action(self, action, steps=1) -> None:
            import numpy as np

            self.policy_calls.append((list(action), steps))
            self.last_osc_action = np.array(action, dtype=np.float32)
            self.move_ee(action[0], action[1], action[2], steps=steps)

    monkeypatch.setenv("VLA_DEBUG_ACTION_TELEMETRY", "1")
    arm = _LiberoArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        axis_remap=None,
        axis_signs=None,
        adaptive_scale=False,
        grasp_dist=0.0,
        max_inner_steps=1,
        sim_pickcube_gripper_profile=False,
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

    with caplog.at_level(logging.INFO, logger="vla.tools"):
        with patch(
            "vla.client.requests.post",
            return_value=_make_response(
                [[0.5, 0.0, -0.3, 0.1, 0.0, 0.0, -1.0]]
            ),
        ):
            tool.invoke({})

    telem = [r.message for r in caplog.records if "VLA telem step=" in r.message]
    assert len(telem) == 2
    assert "agent_state=" in telem[0]
    assert "raw=" in telem[1] and "osc=" in telem[1] and "eef_d=" in telem[1]


def test_vla_state_debug_logs_parity_at_step_zero(monkeypatch, caplog):
    class _LiberoArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__(eef_xyz=[0.0, 0.0, 0.5], obj_xyz=[0.1, 0.0, 0.448])
            self._pos_scale = 1.0
            self._gripper_value = 1.0
            self.last_osc_action = None
            self.last_obs = {
                "agent_pos": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, -1.0],
                "robot_state": {
                    "eef": {
                        "pos": [0.1, 0.2, 0.3],
                        "quat": [0.0, 0.0, 0.0, 1.0],
                    },
                    "gripper": {"qpos": [1.0, -1.0]},
                },
            }

        @property
        def agent_state(self):
            import numpy as np

            return np.array(
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, -1.0],
                dtype=np.float32,
            )

        def apply_policy_action(self, action, steps=1) -> None:
            import numpy as np

            self.last_osc_action = np.array(action, dtype=np.float32)
            self.move_ee(action[0], action[1], action[2], steps=steps)

        def maybe_log_vla_state_parity(self, step: int) -> None:
            from sim.libero_backend import (
                _libero_vla_state_debug_enabled,
                _log_agent_state_source_comparison,
            )

            if step != 0 or not _libero_vla_state_debug_enabled():
                return
            if self.last_obs is not None:
                _log_agent_state_source_comparison(self.last_obs)

    monkeypatch.setenv("LIBERO_VLA_STATE_DEBUG", "1")
    arm = _LiberoArm()
    camera = _FakeCamera()
    cfg = VLAConfig(
        action_scale=1.0,
        axis_remap=None,
        axis_signs=None,
        adaptive_scale=False,
        grasp_dist=0.0,
        max_inner_steps=1,
        sim_pickcube_gripper_profile=False,
    )
    tool = create_vla_manipulation(
        tool_name="pick_cube_state_debug",
        tool_description="test",
        task_prompt="pick the cube",
        server_address="http://localhost:8765",
        arm=arm,
        camera=camera,
        max_steps=1,
        cfg=cfg,
    )

    with caplog.at_level(logging.INFO):
        with patch(
            "vla.client.requests.post",
            return_value=_make_response(
                [[0.5, 0.0, -0.3, 0.1, 0.0, 0.0, -1.0]]
            ),
        ):
            tool.invoke({})

    parity = [r.message for r in caplog.records if "LIBERO state parity" in r.message]
    assert len(parity) == 1
    assert "eef_axis_angle=" in parity[0]
    assert "agent_pos=" in parity[0]

