from __future__ import annotations

from typing import Any

from core.skills import Skill, SkillParam, SkillRegistry


class ArmGrabSkill(Skill):
    name = "arm.grab"
    description = "Pick up a named object using the VLA policy."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to grab"),
    ]
    returns = "str: outcome message from the VLA sequence"

    def __init__(self, vla_tool: Any) -> None:
        # vla_tool: callable that accepts keyword arg `task` (str) and returns str
        self._vla_tool = vla_tool

    def __call__(self, **kwargs: Any) -> str:
        object_name = str(kwargs.get("object_name", "object"))
        return self._vla_tool(task=f"pick up the {object_name}")


class ArmPlaceSkill(Skill):
    name = "arm.place"
    description = "Place a named object at a described location using the VLA policy."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to place"),
        SkillParam("location", "str", "Description of the target location"),
    ]
    returns = "str: outcome message from the VLA sequence"

    def __init__(self, vla_tool: Any) -> None:
        self._vla_tool = vla_tool

    def __call__(self, **kwargs: Any) -> str:
        object_name = str(kwargs.get("object_name", "object"))
        location = str(kwargs.get("location", "target location"))
        return self._vla_tool(task=f"place the {object_name} at {location}")


def register_arm_skills(registry: SkillRegistry, vla_tool: Any) -> None:
    """Register arm.grab and arm.place into *registry*.

    *vla_tool* must be a callable accepting keyword arg `task` (str).
    """
    registry.register(ArmGrabSkill(vla_tool))
    registry.register(ArmPlaceSkill(vla_tool))
