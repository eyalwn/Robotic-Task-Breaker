"""Skill catalog for linear manipulation plans."""
from __future__ import annotations

_SKILL_CATALOG = """\
  open(target_name: str) -> None
    Open a named drawer or door using the VLA policy instruction.
    :param target_name: Drawer or receptacle from the scene snapshot or goal"""


def skill_catalog() -> str:
    """Static catalog text for executor-handled manipulation skills."""
    return _SKILL_CATALOG
