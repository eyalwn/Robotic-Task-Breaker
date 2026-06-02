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
    register_libero_skills,
)


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


def test_list_objects_no_sim():
    vec_env = types.SimpleNamespace(envs=[types.SimpleNamespace(env=None)])
    skill = LiberoListObjectsSkill(vec_env)
    assert skill() == []
    assert _unwrap_to_sim(vec_env) is None
