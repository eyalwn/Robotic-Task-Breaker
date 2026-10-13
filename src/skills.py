"""Skill catalog for linear manipulation plans."""
from __future__ import annotations

_SKILL_CATALOG = """\
  open(object_name: str) -> None
    Open a named drawer or door using the VLA policy instruction.
    :param object_name: Drawer or receptacle from the scene snapshot or goal
  place(object_name: str, source_name: str, destination_name: str) -> None
    Pick an object from a source and place it on a destination.
    :param object_name: Manipulable object to move
    :param source_name: Where the object starts (e.g. counter)
    :param destination_name: Target fixture (e.g. a stove burner phrase)
  put_in(object_name: str, source_name: str, container_name: str) -> None
    Pick an object from a source and place it inside a container receptacle.
    :param object_name: Manipulable object to move
    :param source_name: Where the object starts (e.g. a drawer)
    :param container_name: In-scene receptacle (e.g. bowl)
  close(object_name: str) -> None
    Close a named drawer or door using the VLA policy instruction.
    :param object_name: Drawer or door from the scene snapshot or goal
  turn_on(target_name: str) -> None
    Turn on a named control (e.g. a stove burner).
    :param target_name: Control to activate; use the same burner phrase as place
"""


def skill_catalog() -> str:
    """Static catalog text for executor-handled manipulation skills."""
    return _SKILL_CATALOG
