from __future__ import annotations

from unittest.mock import MagicMock, patch

from vla import VLAConfig, create_vla_manipulation


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

