from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Pose:
    x: float
    y: float
    z: float


@dataclass
class DetectedObject:
    name: str
    pose: Pose
    confidence: float = 1.0


@dataclass
class SkillParam:
    name: str
    type_hint: str
    description: str
    required: bool = True


class Skill(ABC):
    name: str = ""
    description: str = ""
    parameters: list[SkillParam] = []
    returns: str = "None"

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        ...


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._aliases: dict[str, str] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def register_alias(self, alias: str, target: str) -> None:
        if target not in self._skills:
            raise KeyError(f"Skill {target!r} not registered. Available: {sorted(self._skills)}")
        self._aliases[alias] = target

    def get(self, name: str) -> Skill:
        name = self._aliases.get(name, name)
        try:
            return self._skills[name]
        except KeyError:
            available = sorted(set(self._skills) | set(self._aliases))
            raise KeyError(f"Skill {name!r} not registered. Available: {available}")

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def render_catalog(self) -> str:
        """Render a human-readable skill catalog (for use in LLM prompts at M6)."""
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
