"""Skill catalog for linear manipulation plans."""
from __future__ import annotations

_SKILL_CATALOG = """\
  open(object_name: str) -> None
    Open a named drawer or door using the VLA policy instruction.
    :param object_name: Drawer or receptacle from the scene snapshot or goal
  close(object_name: str) -> None
    Close a named drawer or door using the VLA policy instruction.
    :param object_name: Drawer or door from the scene snapshot or goal
"""


def skill_catalog() -> str:
    """Static catalog text for executor-handled manipulation skills."""
    return _SKILL_CATALOG
