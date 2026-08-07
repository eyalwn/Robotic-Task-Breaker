"""Skill registry and arm-skill catalog for plan execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SkillParam:
    """Catalog metadata for one skill parameter."""

    def __init__(
        self,
        name: str,
        type_hint: str,
        description: str,
        required: bool = True,
    ) -> None:
        self.name = name
        self.type_hint = type_hint
        self.description = description
        self.required = required


class Skill(ABC):
    """Callable plan skill with catalog metadata."""

    name: str = ""
    description: str = ""
    parameters: list[SkillParam] = []
    returns: str = "None"

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        ...


class SkillRegistry:
    """Registry of plan skills keyed by name."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._aliases: dict[str, str] = {}

    def register(self, skill: Skill) -> None:
        """Add a skill to the registry."""
        self._skills[skill.name] = skill

    def register_alias(self, alias: str, target: str) -> None:
        """Map an alternate name to a registered skill."""
        if target not in self._skills:
            raise KeyError(f"Skill {target!r} not registered. Available: {sorted(self._skills)}")
        self._aliases[alias] = target

    def get(self, name: str) -> Skill:
        """Resolve a skill by name or alias."""
        name = self._aliases.get(name, name)
        try:
            return self._skills[name]
        except KeyError:
            available = sorted(set(self._skills) | set(self._aliases))
            raise KeyError(f"Skill {name!r} not registered. Available: {available}")

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def render_catalog(self) -> str:
        """Format registered skills for planner prompts."""
        lines: list[str] = []
        for skill in self._skills.values():
            params = ", ".join(
                f"{p.name}: {p.type_hint}" + ("" if p.required else "?")
                for p in skill.parameters
            )
            lines.append(f"  {skill.name}({params}) -> {skill.returns}")
            lines.append(f"    {skill.description}")
            for p in skill.parameters:
                lines.append(f"    :param {p.name}: {p.description}")
        return "\n".join(lines)


_ARM_CATALOG = """\
  arm.grab(object_name: str) -> dict
    Pick up a named object using the VLA manipulation policy.
    :param object_name: Object name from the scene snapshot or goal (e.g. alphabet soup)
  arm.place(object_name: str, location: str) -> dict
    Place a held object at a location using the VLA manipulation policy.
    :param object_name: Object name from the scene snapshot or goal
    :param location: Target location name (e.g. basket)
  arm.transfer(object_name: str, location: str) -> dict
    Move a named object to a location using the VLA manipulation policy.
    :param object_name: Object name from the scene snapshot or goal
    :param location: Target location name
  arm.open(target_name: str) -> dict
    Open a named target (e.g. drawer, door) using the VLA manipulation policy.
    :param target_name: Target name from the scene snapshot or goal
  arm.close(target_name: str) -> dict
    Close a named target using the VLA manipulation policy.
    :param target_name: Target name from the scene snapshot or goal
  arm.push(target_name: str, destination_or_direction: str) -> dict
    Push a named object toward a destination or direction using the VLA policy.
    :param target_name: Object name from the scene snapshot or goal
    :param destination_or_direction: Destination location or push direction
  arm.pull(target_name: str, destination_or_direction: str) -> dict
    Pull a named object toward a destination or direction using the VLA policy.
    :param target_name: Object name from the scene snapshot or goal
    :param destination_or_direction: Destination location or pull direction"""


def arm_skill_catalog() -> str:
    """Static catalog text for executor-handled arm skills."""
    return _ARM_CATALOG
