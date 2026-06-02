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

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(f"Skill {name!r} not registered. Available: {sorted(self._skills)}")

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
