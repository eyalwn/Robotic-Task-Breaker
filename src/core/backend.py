from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.skills import SkillRegistry


@runtime_checkable
class Backend(Protocol):
    def register_skills(self, registry: SkillRegistry, goal: str) -> None: ...

    def capture_image(self) -> bytes: ...

    def success(self) -> bool: ...

    def close(self) -> None: ...
