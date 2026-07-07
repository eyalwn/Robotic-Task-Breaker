from __future__ import annotations

import pytest

from core.skills import SkillRegistry
from vla.arm_skills import (
    ArmGrabSkill,
    ArmPlaceSkill,
    _place_task_prompt,
    register_arm_skills,
)


class MockVlaTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return "ok"


def test_arm_grab_skill_default_uses_raw_object_name() -> None:
    vla_tool = MockVlaTool()
    skill = ArmGrabSkill(vla_tool)
    skill(object_name="alphabet_soup_1_main")
    assert vla_tool.calls == [{"task": "pick up the alphabet_soup_1_main"}]


def test_arm_grab_skill_with_name_mapper() -> None:
    from sim.libero_skills import libero_object_human_name

    vla_tool = MockVlaTool()
    skill = ArmGrabSkill(vla_tool, name_mapper=libero_object_human_name)
    skill(object_name="alphabet_soup_1_main")
    assert vla_tool.calls == [{"task": "pick up the alphabet soup can"}]


def test_arm_place_skill_default_uses_raw_object_name() -> None:
    vla_tool = MockVlaTool()
    skill = ArmPlaceSkill(vla_tool)
    skill(object_name="cream_cheese_1_main", location="basket_1_main")
    assert vla_tool.calls == [
        {"task": "put the cream_cheese_1_main in the basket"}
    ]


def test_arm_place_skill_basket_location_prompt() -> None:
    vla_tool = MockVlaTool()
    skill = ArmPlaceSkill(vla_tool)
    skill(object_name="alphabet_soup_1_main", location="basket")
    assert vla_tool.calls == [
        {"task": "put the alphabet_soup_1_main in the basket"}
    ]
    assert _place_task_prompt("alphabet soup can", "basket") == (
        "put the alphabet soup can in the basket"
    )


def test_arm_place_skill_default_returns_raw_vla_result() -> None:
    stagnation = '{"status":"stopped","reason":"stagnation"}'

    class _StagnationVla:
        def __call__(self, **_kwargs: object) -> str:
            return stagnation

    class _Arm:
        def set_target_object(self, name: str) -> bool:
            return True

    raw = ArmPlaceSkill(_StagnationVla(), arm=_Arm())(
        object_name="alphabet_soup_1_main", location="basket"
    )
    assert raw == stagnation


def test_arm_grab_skill_sets_target_object() -> None:
    class _Arm:
        def __init__(self) -> None:
            self.target: str | None = None

        def set_target_object(self, name: str) -> bool:
            self.target = name
            return True

    vla_tool = MockVlaTool()
    arm = _Arm()
    skill = ArmGrabSkill(vla_tool, arm=arm)
    skill(object_name="alphabet_soup_1")
    assert arm.target == "alphabet_soup_1"


def test_register_arm_skills_registers_both() -> None:
    registry = SkillRegistry()
    register_arm_skills(registry, MockVlaTool())
    assert registry.get("arm.grab")
    assert registry.get("arm.place")


def test_arm_place_skill_sets_basket_target_when_location_mentions_basket() -> None:
    class _Arm:
        def __init__(self) -> None:
            self.target: str | None = None

        def set_target_object(self, name: str) -> bool:
            self.target = name
            return True

    vla_tool = MockVlaTool()
    arm = _Arm()
    skill = ArmPlaceSkill(vla_tool, arm=arm)
    skill(object_name="alphabet_soup_1_main", location="in the basket")
    assert arm.target == "basket_1_main"
    assert vla_tool.calls == [
        {"task": "put the alphabet_soup_1_main in the basket"}
    ]


def test_arm_place_skill_keeps_object_target_for_non_basket_location() -> None:
    class _Arm:
        def __init__(self) -> None:
            self.target: str | None = None

        def set_target_object(self, name: str) -> bool:
            self.target = name
            return True

    vla_tool = MockVlaTool()
    arm = _Arm()
    skill = ArmPlaceSkill(vla_tool, arm=arm)
    skill(object_name="cream_cheese_1_main", location="on the table")
    assert arm.target == "cream_cheese_1_main"


def test_arm_grab_prepares_after_place(monkeypatch) -> None:
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_AFTER_PLACE", "1")

    class _Arm:
        def __init__(self) -> None:
            self.target: str | None = None
            self.prepared = 0
            self.prep_next: str | None = None

        def set_target_object(self, name: str) -> bool:
            self.target = name
            self._target_object = name
            return True

        def prepare_for_grab(self, next_object: str | None = None) -> None:
            self.prepared += 1
            self.prep_next = next_object

    vla_tool = MockVlaTool()
    arm = _Arm()
    place = ArmPlaceSkill(vla_tool, arm=arm)
    grab = ArmGrabSkill(vla_tool, arm=arm)
    place(object_name="alphabet_soup_1_main", location="in the basket")
    grab(object_name="cream_cheese_1_main")
    assert arm.prepared == 1
    assert arm.target == "cream_cheese_1_main"
    assert arm.prep_next == "cream_cheese_1_main"


def test_arm_grab_skips_prepare_when_not_after_place(monkeypatch) -> None:
    monkeypatch.setenv("LIBERO_PREPARE_GRAB_AFTER_PLACE", "1")

    class _Arm:
        prepared = 0

        def set_target_object(self, name: str) -> bool:
            return True

        def prepare_for_grab(self, next_object: str | None = None) -> None:
            self.prepared += 1

    vla_tool = MockVlaTool()
    arm = _Arm()
    ArmGrabSkill(vla_tool, arm=arm)(object_name="alphabet_soup_1_main")
    assert arm.prepared == 0


def test_missing_skill_raises_keyerror() -> None:
    registry = SkillRegistry()
    with pytest.raises(KeyError):
        registry.get("arm.grab")
