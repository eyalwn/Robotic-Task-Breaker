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


def test_arm_grab_skill_calls_vla() -> None:
    vla_tool = MockVlaTool()
    skill = ArmGrabSkill(vla_tool)
    skill(object_name="alphabet_soup_1_main")
    assert vla_tool.calls == [{"task": "pick up the alphabet soup can"}]


def test_arm_place_skill_calls_vla() -> None:
    vla_tool = MockVlaTool()
    skill = ArmPlaceSkill(vla_tool)
    skill(object_name="cream_cheese_1_main", location="basket_1_main")
    assert vla_tool.calls == [
        {"task": "put the cream cheese box in the basket"}
    ]


def test_arm_place_skill_basket_location_emits_libero_prompt() -> None:
    vla_tool = MockVlaTool()
    skill = ArmPlaceSkill(vla_tool)
    skill(object_name="alphabet_soup_1_main", location="basket")
    assert vla_tool.calls == [
        {"task": "put the alphabet soup can in the basket"}
    ]
    assert _place_task_prompt("alphabet soup can", "basket") == (
        "put the alphabet soup can in the basket"
    )


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
        {"task": "put the alphabet soup can in the basket"}
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


def test_arm_place_skill_upgrades_stagnation_when_near_basket(monkeypatch) -> None:
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

    raw = ArmPlaceSkill(_StagnationVla(), arm=_Arm())(
        object_name="alphabet_soup_1_main", location="basket"
    )
    import json

    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload["reason"] == "place_proximity"
    assert payload["place_xy_dist"] == 0.05


def test_arm_place_skill_keeps_stagnation_when_far_from_basket(monkeypatch) -> None:
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

    class _StagnationVla:
        def __call__(self, **_kwargs: object) -> str:
            return stagnation

    raw = ArmPlaceSkill(_StagnationVla(), arm=_Arm())(
        object_name="alphabet_soup_1_main", location="basket"
    )
    import json

    assert json.loads(raw)["reason"] == "stagnation"


def test_missing_skill_raises_keyerror() -> None:
    registry = SkillRegistry()
    with pytest.raises(KeyError):
        registry.get("arm.grab")
