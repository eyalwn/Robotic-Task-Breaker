from __future__ import annotations

import types

import numpy as np

from core.skills import DetectedObject, Pose, SkillRegistry
from sim.maniskill_skills import (
    ManiSkillDetectSkill,
    ManiSkillListObjectsSkill,
    ManiSkillLocateSkill,
    register_maniskill_skills,
)


def _make_actor(xyz):
    p = np.array(xyz, dtype=np.float32)
    pose = types.SimpleNamespace(p=p)
    return types.SimpleNamespace(pose=pose)


def _make_fake_env(actors: dict):
    unwrapped_dict = {name: _make_actor(xyz) for name, xyz in actors.items()}
    unwrapped_dict["agent"] = object()
    unwrapped = types.SimpleNamespace(**unwrapped_dict)
    return types.SimpleNamespace(unwrapped=unwrapped)


def test_list_objects_excludes_agent():
    env = _make_fake_env({"cube": [0.1, 0.2, 0.3]})
    skill = ManiSkillListObjectsSkill(env)
    assert skill() == ["cube"]


def test_detect_cube():
    env = _make_fake_env({"cube": [1.0, 2.0, 3.0]})
    skill = ManiSkillDetectSkill(env)
    result = skill(object_name="cube")
    assert isinstance(result, DetectedObject)
    assert result.name == "cube"
    assert result.pose == Pose(x=1.0, y=2.0, z=3.0)


def test_detect_missing():
    env = _make_fake_env({"cube": [0.0, 0.0, 0.0]})
    skill = ManiSkillDetectSkill(env)
    assert skill(object_name="missing") is None


def test_locate_cube():
    env = _make_fake_env({"cube": [0.5, 0.6, 0.7]})
    skill = ManiSkillLocateSkill(env)
    result = skill(object_name="cube")
    assert isinstance(result, Pose)
    assert result == Pose(x=0.5, y=0.6, z=0.7)


def test_locate_missing():
    env = _make_fake_env({"cube": [0.0, 0.0, 0.0]})
    skill = ManiSkillLocateSkill(env)
    assert skill(object_name="missing") is None


def test_register_maniskill_skills():
    registry = SkillRegistry()
    register_maniskill_skills(registry, _make_fake_env({}))
    assert registry.get("state.list_objects")
    assert registry.get("perception.detect")
    assert registry.get("perception.locate")
