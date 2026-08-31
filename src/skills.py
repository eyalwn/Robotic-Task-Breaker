"""Skill catalog for libero_10 linear plans."""
from __future__ import annotations

_SKILL_CATALOG = """\
  grab(object_name: str) -> None
    Pick up a named object using the VLA manipulation policy.
    :param object_name: Object name from the scene snapshot or goal
  place(object_name: str, target_name: str, position: str) -> None
    Place a held object relative to a target using the VLA manipulation policy.
    :param object_name: Object name from the scene snapshot or goal
    :param target_name: Target name (e.g. basket, plate, drawer region)
    :param position: One of center, left, right, top, bottom, inside, on
  close(target_name: str) -> None
    Close a named target (e.g. drawer, microwave) using the VLA manipulation policy.
    :param target_name: Target name from the scene snapshot or goal
  turn_on(target_name: str) -> None
    Turn on a named appliance or switch using the VLA manipulation policy.
    :param target_name: Target name from the scene snapshot or goal"""


def skill_catalog() -> str:
    """Static catalog text for executor-handled manipulation skills."""
    return _SKILL_CATALOG
