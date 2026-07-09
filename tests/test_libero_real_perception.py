from __future__ import annotations

import types

import numpy as np
import pytest

from core.skills import DetectedObject, Pose, SkillRegistry
from sim.libero_perception import (
    PERCEPTION_MODES,
    register_libero_perception_skills,
    resolve_perception_backend,
)
from sim.libero_real_perception import (
    RealDetectSkill,
    RealListObjectsSkill,
    RealLocateSkill,
    _normalize_object_name,
    _pixel_to_table_pose,
    detect_objects_in_frame,
    register_libero_real_perception,
)


def _solid_frame(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _blob_frame(
    h: int,
    w: int,
    *,
    color: tuple[int, int, int],
    center: tuple[int, int],
    radius: int = 18,
) -> np.ndarray:
    import cv2

    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.circle(frame, center, radius, color, thickness=-1)
    return frame


def test_resolve_perception_backend_defaults_to_sim_gt(monkeypatch):
    monkeypatch.delenv("PERCEPTION_BACKEND", raising=False)
    assert resolve_perception_backend() == "sim_gt"


@pytest.mark.parametrize("mode", PERCEPTION_MODES)
def test_resolve_perception_backend_accepts_known_modes(monkeypatch, mode):
    monkeypatch.setenv("PERCEPTION_BACKEND", mode)
    assert resolve_perception_backend() == mode


def test_resolve_perception_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv("PERCEPTION_BACKEND", "oracle")
    with pytest.raises(ValueError, match="PERCEPTION_BACKEND"):
        resolve_perception_backend()


def test_normalize_object_name_aliases():
    assert _normalize_object_name("alphabet soup can") == "alphabet_soup_1_main"
    assert _normalize_object_name("cream cheese box") == "cream_cheese_1_main"
    assert _normalize_object_name("basket") == "basket_1_main"
    assert _normalize_object_name("unknown widget") is None


def test_detect_objects_in_frame_finds_colored_blobs():
    frame = _blob_frame(
        128,
        128,
        color=(220, 40, 40),
        center=(40, 60),
    )
    hits = detect_objects_in_frame(frame, object_names=("alphabet_soup_1_main",))
    assert len(hits) == 1
    assert hits[0].body_name == "alphabet_soup_1_main"
    assert hits[0].area > 0


def test_real_detect_skill_from_injected_frame():
    frame = _blob_frame(
        160,
        160,
        color=(220, 40, 40),
        center=(80, 80),
    )
    skill = RealDetectSkill(
        camera=None,
        frame_fn=lambda: frame,
    )
    result = skill(object_name="alphabet soup can")
    assert isinstance(result, DetectedObject)
    assert result.name == "alphabet_soup_1_main"
    assert isinstance(result.pose, Pose)


def test_real_locate_skill_returns_pose():
    frame = _blob_frame(
        160,
        160,
        color=(30, 120, 200),
        center=(50, 100),
    )
    skill = RealLocateSkill(camera=None, frame_fn=lambda: frame)
    pose = skill(object_name="cream cheese box")
    assert isinstance(pose, Pose)


def test_real_list_objects_returns_detections_or_empty():
    frame = _blob_frame(
        160,
        160,
        color=(220, 40, 40),
        center=(80, 80),
    )
    skill = RealListObjectsSkill(camera=None, frame_fn=lambda: frame)
    assert "alphabet_soup_1_main" in skill()

    empty_skill = RealListObjectsSkill(
        camera=None, frame_fn=lambda: _solid_frame(32, 32, (5, 5, 5))
    )
    assert empty_skill() == []


def test_pixel_to_table_pose_is_deterministic():
    pose = _pixel_to_table_pose(120.0, 20.0, 160, 80)
    assert pose.x == pytest.approx(0.22 * 0.5)
    assert pose.y == pytest.approx(-0.18 * -0.5)
    assert pose.z == pytest.approx(0.92)


def test_register_libero_real_perception_marks_real_source():
    registry = SkillRegistry()
    register_libero_real_perception(registry, camera=object())
    for name in ("state.list_objects", "perception.detect", "perception.locate"):
        skill = registry.get(name)
        assert getattr(skill, "perception_source", None) == "real"


def test_register_libero_perception_skills_sim_gt_uses_sim_gt_wiring(monkeypatch):
    monkeypatch.setenv("PERCEPTION_BACKEND", "sim_gt")
    registry = SkillRegistry()
    vec_env = types.SimpleNamespace()
    camera = types.SimpleNamespace()

    called: list[str] = []

    def _fake_register(registry_obj, env):
        called.append("sim_gt")
        assert env is vec_env

    monkeypatch.setattr(
        "sim.libero_skills.register_libero_skills",
        _fake_register,
    )

    mode = register_libero_perception_skills(
        registry, vec_env=vec_env, camera=camera
    )
    assert mode == "sim_gt"
    assert called == ["sim_gt"]


def test_register_libero_perception_skills_real_avoids_sim_gt(monkeypatch):
    monkeypatch.setenv("PERCEPTION_BACKEND", "real")
    registry = SkillRegistry()
    vec_env = types.SimpleNamespace()
    camera = types.SimpleNamespace()

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("sim-GT perception must not register in real mode")

    monkeypatch.setattr(
        "sim.libero_skills.register_libero_skills",
        _forbidden,
    )

    mode = register_libero_perception_skills(
        registry, vec_env=vec_env, camera=camera
    )
    assert mode == "real"
    assert registry.get("perception.detect").perception_source == "real"
    assert registry.get("state.list_objects").perception_source == "real"


def test_libero_backend_real_mode_guard_against_sim_gt_fallback(monkeypatch):
    import vla.arm_skills as arm_skills_mod

    from sim.libero_backend import LiberoBackend

    monkeypatch.setenv("PERCEPTION_BACKEND", "real")

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("LiberoBackend real mode routed to sim-GT perception")

    monkeypatch.setattr(
        "sim.libero_skills.register_libero_skills",
        _forbidden,
    )

    env = types.SimpleNamespace(close=lambda: None)
    arm = types.SimpleNamespace(_env=env, last_obs=None)
    camera = types.SimpleNamespace(
        capture_image=lambda: b"",
        is_placeholder_jpeg=lambda _b: True,
    )

    backend = LiberoBackend.__new__(LiberoBackend)
    backend._env = env
    backend._arm = arm
    backend._camera = camera
    backend._vla_port = 8765

    monkeypatch.setattr(
        arm_skills_mod,
        "register_arm_skills",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vla.create_vla_manipulation",
        lambda **kwargs: types.SimpleNamespace(invoke=lambda _a: "ok"),
    )
    monkeypatch.setattr(
        "vla.VLAConfig.from_env",
        lambda: type("Cfg", (), {"max_inner_steps": 10})(),
    )

    registry = SkillRegistry()
    backend.register_skills(registry, "put objects in basket")

    assert backend._perception_mode == "real"
    detect = registry.get("perception.detect")
    assert detect.perception_source == "real"
    assert type(detect).__module__ == "sim.libero_real_perception"
