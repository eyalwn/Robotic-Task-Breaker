from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from libero_env import LiberoEnv


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


class ListObjectsSkill(Skill):
    name = "state.list_objects"
    description = "Return the names of manipulable objects in the current LIBERO scene."
    parameters: list[SkillParam] = []
    returns = "list[str]: object names"

    def __init__(self, env: LiberoEnv) -> None:
        self._env = env

    def __call__(self, **kwargs: Any) -> list[str]:
        return self._env.object_names()


class DetectSkill(Skill):
    name = "perception.detect"
    description = "Detect a named object in the LIBERO scene and return its pose."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to detect"),
    ]
    returns = "DetectedObject | None"

    def __init__(self, env: LiberoEnv) -> None:
        self._env = env

    def __call__(self, **kwargs: Any) -> DetectedObject | None:
        object_name = str(kwargs.get("object_name", ""))
        xyz = self._env.body_position(object_name)
        if xyz is None:
            return None
        return DetectedObject(
            name=object_name,
            pose=Pose(x=xyz[0], y=xyz[1], z=xyz[2]),
        )


class LocateSkill(Skill):
    name = "perception.locate"
    description = "Return the 3-D position of a named object in the LIBERO scene."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to locate"),
    ]
    returns = "Pose | None"

    def __init__(self, env: LiberoEnv) -> None:
        self._detect = DetectSkill(env)

    def __call__(self, **kwargs: Any) -> Pose | None:
        result = self._detect(object_name=kwargs.get("object_name", ""))
        return result.pose if result is not None else None


def register_libero_skills(registry: SkillRegistry, env: LiberoEnv) -> None:
    registry.register(ListObjectsSkill(env))
    registry.register_alias("list_objects", "state.list_objects")
    registry.register(DetectSkill(env))
    registry.register(LocateSkill(env))


_ARM_CATALOG = """\
  arm.grab(object_name: str) -> dict
    Pick up a named object using the VLA manipulation policy.
    :param object_name: BDDL object id (e.g. alphabet_soup_1)
  arm.place(object_name: str, location: str) -> dict
    Place a named object at a location using the VLA manipulation policy.
    :param object_name: BDDL object id (e.g. alphabet_soup_1)
    :param location: Target location name (e.g. basket)"""


def arm_skill_catalog() -> str:
    """Static catalog text for executor-handled arm skills."""
    return _ARM_CATALOG
