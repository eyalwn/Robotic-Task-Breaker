from __future__ import annotations

import types

import numpy as np
import pytest

from core.skills import DetectedObject, Pose, SkillRegistry
from sim.libero_skills import (
    LiberoDetectSkill,
    LiberoListObjectsSkill,
    LiberoLocateSkill,
    _unwrap_to_sim,
    libero_object_human_name,
    libero_place_result_hook,
    register_libero_skills,
)
from vla.arm_skills import ArmPlaceSkill


def _make_fake_sim(body_names, body_positions):
    """body_positions: dict[name, [x,y,z]]"""
    name2id = {n: i for i, n in enumerate(body_names)}
    xpos = np.array([body_positions.get(n, [0.0, 0.0, 0.0]) for n in body_names])
    model = types.SimpleNamespace(
        body_names=body_names,
        body_name2id=lambda n: name2id[n],
    )
    data = types.SimpleNamespace(body_xpos=xpos)
    return types.SimpleNamespace(model=model, data=data)


def _make_fake_vec_env(sim):
    inner = types.SimpleNamespace(sim=sim)
    outer = types.SimpleNamespace(env=inner)
    return types.SimpleNamespace(envs=[outer])


def test_libero_object_human_name_alias():
    assert libero_object_human_name("alphabet_soup_1_main") == "alphabet soup can"


def test_libero_object_human_name_fallback_strip_main():
    assert libero_object_human_name("butter_1_main") == "butter"


def test_list_objects_filters_robot_bodies():
    sim = _make_fake_sim(
        ["worldBody", "robot0_base", "akita_black_bowl_1", "butter_1"],
        {
            "akita_black_bowl_1": [0.1, 0.2, 0.3],
            "butter_1": [0.4, 0.5, 0.6],
        },
    )
    skill = LiberoListObjectsSkill(_make_fake_vec_env(sim))
    assert skill() == ["akita_black_bowl_1", "butter_1"]


def test_detect_known_object():
    sim = _make_fake_sim(
        ["akita_black_bowl_1"],
        {"akita_black_bowl_1": [1.0, 2.0, 3.0]},
    )
    skill = LiberoDetectSkill(_make_fake_vec_env(sim))
    result = skill(object_name="akita_black_bowl_1")
    assert isinstance(result, DetectedObject)
    assert result.name == "akita_black_bowl_1"
    assert result.pose == Pose(x=1.0, y=2.0, z=3.0)


def test_detect_unknown_object():
    sim = _make_fake_sim(["akita_black_bowl_1"], {"akita_black_bowl_1": [0.0, 0.0, 0.0]})
    skill = LiberoDetectSkill(_make_fake_vec_env(sim))
    assert skill(object_name="nonexistent") is None


def test_locate_returns_pose():
    sim = _make_fake_sim(
        ["butter_1"],
        {"butter_1": [0.5, 0.6, 0.7]},
    )
    skill = LiberoLocateSkill(_make_fake_vec_env(sim))
    result = skill(object_name="butter_1")
    assert isinstance(result, Pose)
    assert result == Pose(x=0.5, y=0.6, z=0.7)


def test_locate_unknown_returns_none():
    sim = _make_fake_sim(["butter_1"], {"butter_1": [0.0, 0.0, 0.0]})
    skill = LiberoLocateSkill(_make_fake_vec_env(sim))
    assert skill(object_name="nonexistent") is None


def test_register_libero_skills():
    registry = SkillRegistry()
    register_libero_skills(registry, _make_fake_vec_env(_make_fake_sim([], {})))
    assert registry.get("state.list_objects")
    assert registry.get("perception.detect")
    assert registry.get("perception.locate")


def test_list_objects_alias():
    sim = _make_fake_sim(
        ["akita_black_bowl_1"],
        {"akita_black_bowl_1": [0.1, 0.2, 0.3]},
    )
    registry = SkillRegistry()
    register_libero_skills(registry, _make_fake_vec_env(sim))
    assert registry.get("list_objects") is registry.get("state.list_objects")
    assert registry.get("list_objects")() == ["akita_black_bowl_1"]


def test_list_objects_no_sim():
    vec_env = types.SimpleNamespace(envs=[types.SimpleNamespace(env=None)])
    skill = LiberoListObjectsSkill(vec_env)
    assert skill() == []
    assert _unwrap_to_sim(vec_env) is None


def test_libero_place_result_hook_upgrades_stagnation_when_near_basket(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIBERO_PLACE_SMOKE_MAX_XY", "0.12")

    class _Env:
        pass

    class _Arm:
        def __init__(self) -> None:
            self._env = _Env()

    positions = {
        "alphabet_soup_1_main": [0.0, 0.0, 0.5],
        "basket_1_main": [0.05, 0.0, 0.5],
    }
    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        lambda _env, body: positions.get(body),
    )

    stagnation = (
        '{"status":"stopped","reason":"stagnation","steps_applied":12,'
        '"min_dist":0.08,"grasp_triggered":false}'
    )
    raw = libero_place_result_hook(
        stagnation, _Arm(), "alphabet_soup_1_main", "basket"
    )
    import json

    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload["reason"] == "place_proximity"
    assert payload["place_xy_dist"] == 0.05
    assert payload["grasp_triggered"] is True


def test_libero_place_result_hook_keeps_stagnation_when_far(monkeypatch) -> None:
    monkeypatch.setenv("LIBERO_PLACE_SMOKE_MAX_XY", "0.12")

    class _Env:
        pass

    class _Arm:
        _env = _Env()

    positions = {
        "alphabet_soup_1_main": [0.5, 0.0, 0.5],
        "basket_1_main": [0.0, 0.0, 0.5],
    }
    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        lambda _env, body: positions.get(body),
    )

    stagnation = '{"status":"stopped","reason":"stagnation","steps_applied":12}'
    raw = libero_place_result_hook(
        stagnation, _Arm(), "alphabet_soup_1_main", "basket"
    )
    import json

    assert json.loads(raw)["reason"] == "stagnation"


def test_arm_place_skill_with_libero_hooks(monkeypatch) -> None:
    monkeypatch.setenv("LIBERO_PLACE_SMOKE_MAX_XY", "0.12")

    class _Env:
        pass

    class _Arm:
        def __init__(self) -> None:
            self._env = _Env()

        def set_target_object(self, name: str) -> bool:
            return True

    positions = {
        "alphabet_soup_1_main": [0.0, 0.0, 0.5],
        "basket_1_main": [0.05, 0.0, 0.5],
    }
    monkeypatch.setattr(
        "sim.libero_skills.libero_body_position",
        lambda _env, body: positions.get(body),
    )

    stagnation = (
        '{"status":"stopped","reason":"stagnation","steps_applied":12,'
        '"min_dist":0.08,"grasp_triggered":false}'
    )

    class _StagnationVla:
        def __call__(self, **_kwargs: object) -> str:
            return stagnation

    raw = ArmPlaceSkill(
        _StagnationVla(),
        arm=_Arm(),
        name_mapper=libero_object_human_name,
        place_result_hook=libero_place_result_hook,
    )(object_name="alphabet_soup_1_main", location="basket")
    import json

    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload["reason"] == "place_proximity"


def test_unwrap_to_sim_via_unwrapped():
    """_unwrap_to_sim must find sim via .unwrapped (gymnasium-style chain)."""
    from sim.libero_skills import _unwrap_to_sim

    fake_sim = object()

    class InnerEnv:
        sim = fake_sim

    class OuterEnv:
        # no .env attribute — only gymnasium-style .unwrapped
        unwrapped = InnerEnv()

    class FakeVecEnv:
        envs = [OuterEnv()]

    assert _unwrap_to_sim(FakeVecEnv()) is fake_sim
