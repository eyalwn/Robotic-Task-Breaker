from __future__ import annotations

import pytest

from core.skills import SkillRegistry
from vla.arm_skills import ArmGrabSkill, ArmPlaceSkill, register_arm_skills


class MockVlaTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def __call__(self, **kwargs: str) -> str:
        self.calls.append(dict(kwargs))
        return "ok"


def test_arm_grab_skill_calls_vla() -> None:
    vla_tool = MockVlaTool()
    skill = ArmGrabSkill(vla_tool)
    skill(object_name="cube")
    assert vla_tool.calls == [{"task": "pick up the cube"}]


def test_arm_place_skill_calls_vla() -> None:
    vla_tool = MockVlaTool()
    skill = ArmPlaceSkill(vla_tool)
    skill(object_name="cube", location="table")
    assert vla_tool.calls == [{"task": "place the cube at table"}]


def test_register_arm_skills_registers_both() -> None:
    registry = SkillRegistry()
    register_arm_skills(registry, MockVlaTool())
    assert registry.get("arm.grab")
    assert registry.get("arm.place")


def test_missing_skill_raises_keyerror() -> None:
    registry = SkillRegistry()
    with pytest.raises(KeyError):
        registry.get("arm.grab")
