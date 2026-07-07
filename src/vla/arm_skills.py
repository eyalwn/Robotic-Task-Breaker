from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from core.skills import Skill, SkillParam, SkillRegistry


def _maybe_set_target(arm: Any, object_name: str) -> None:
    setter = getattr(arm, "set_target_object", None)
    if callable(setter):
        setter(object_name)


def _mark_arm_skill(arm: Any, skill: str) -> None:
    if arm is not None:
        setattr(arm, "_longor_last_skill", skill)


def _place_task_prompt(human_name: str, location: str) -> str:
    """LIBERO/OpenPI-native phrasing when the target is a basket."""
    if "basket" in location.lower():
        return f"put the {human_name} in the basket"
    return f"place the {human_name} at {location}"


def _prepare_grab_after_place(arm: Any) -> None:
    if arm is None or getattr(arm, "_longor_last_skill", None) != "place":
        return
    if os.environ.get("LIBERO_PREPARE_GRAB_AFTER_PLACE", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return
    prep = getattr(arm, "prepare_for_grab", None)
    if callable(prep):
        next_object = getattr(arm, "_target_object", None)
        prep(next_object=next_object)


class ArmGrabSkill(Skill):
    name = "arm.grab"
    description = "Pick up a named object using the VLA policy."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to grab"),
    ]
    returns = "str: outcome message from the VLA sequence"

    def __init__(
        self,
        vla_tool: Any,
        arm: Any = None,
        name_mapper: Callable[[str], str] | None = None,
    ) -> None:
        self._vla_tool = vla_tool
        self._arm = arm
        self._name_mapper = name_mapper or (lambda n: n)

    def __call__(self, **kwargs: Any) -> str:
        object_name = str(kwargs.get("object_name", ""))
        if self._arm is not None:
            _maybe_set_target(self._arm, object_name)
            _prepare_grab_after_place(self._arm)
            _mark_arm_skill(self._arm, "grab")
        human_name = self._name_mapper(object_name)
        return self._vla_tool(task=f"pick up the {human_name}")


class ArmPlaceSkill(Skill):
    name = "arm.place"
    description = (
        "Place a named object at a described location using the VLA policy. "
        "When location mentions a basket, proximity tracking targets basket_1_main."
    )
    parameters = [
        SkillParam("object_name", "str", "Name of the object to place"),
        SkillParam("location", "str", "Description of the target location"),
    ]
    returns = "str: outcome message from the VLA sequence"

    def __init__(
        self,
        vla_tool: Any,
        arm: Any = None,
        name_mapper: Callable[[str], str] | None = None,
        place_result_hook: Callable[[str, Any, str, str], str] | None = None,
    ) -> None:
        self._vla_tool = vla_tool
        self._arm = arm
        self._name_mapper = name_mapper or (lambda n: n)
        self._place_result_hook = place_result_hook

    def __call__(self, **kwargs: Any) -> str:
        object_name = str(kwargs.get("object_name", ""))
        location = str(kwargs.get("location", ""))
        if self._arm is not None:
            if "basket" in location.lower():
                _maybe_set_target(self._arm, "basket_1_main")
            else:
                _maybe_set_target(self._arm, object_name)
            _mark_arm_skill(self._arm, "place")
        human_name = self._name_mapper(object_name)
        raw = self._vla_tool(task=_place_task_prompt(human_name, location))
        if self._arm is not None and self._place_result_hook is not None:
            return self._place_result_hook(raw, self._arm, object_name, location)
        return raw


def register_arm_skills(
    registry: SkillRegistry,
    vla_tool: Any,
    arm: Any = None,
    *,
    name_mapper: Callable[[str], str] | None = None,
    place_result_hook: Callable[[str, Any, str, str], str] | None = None,
) -> None:
    """Register arm.grab and arm.place into *registry*.

    *vla_tool* must be a callable accepting keyword arg ``task`` (str).
    Optional *arm* receives ``set_target_object`` before each grab/place.
    """
    registry.register(
        ArmGrabSkill(vla_tool, arm=arm, name_mapper=name_mapper)
    )
    registry.register(
        ArmPlaceSkill(
            vla_tool,
            arm=arm,
            name_mapper=name_mapper,
            place_result_hook=place_result_hook,
        )
    )
