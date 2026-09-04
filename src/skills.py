"""Skill catalog for linear manipulation plans."""
from __future__ import annotations

_SKILL_CATALOG = """\
  transfer(object_name: str, target_name: str) -> None
    Move a named object into a named target receptacle using the VLA policy.
    :param object_name: Object name from the scene snapshot or goal
    :param target_name: Target receptacle name from the scene snapshot or goal"""


def skill_catalog() -> str:
    """Static catalog text for executor-handled manipulation skills."""
    return _SKILL_CATALOG
