"""Reusable plan presets for examples and tests."""
from __future__ import annotations

from .schema import Plan, validate_plan

_FIXTURE_PARTS = ("basket", "cabinet", "plate", "drawer", "bin")


def resolve_libero_grasp_object(
    objects: list[str],
    default: str = "alphabet_soup_1",
) -> str:
    if default in objects or not objects:
        return default
    for name in sorted(objects):
        low = name.lower()
        if any(part in low for part in _FIXTURE_PARTS):
            continue
        return name
    return default


def grab_object_plan(object_name: str, goal: str | None = None) -> Plan:
    resolved_goal = goal if goal is not None else f"pick up {object_name}"
    return validate_plan(
        {
            "goal": resolved_goal,
            "body": {
                "kind": "seq",
                "steps": [
                    {
                        "kind": "call",
                        "skill": "perception.locate",
                        "args": {
                            "object_name": {
                                "kind": "literal",
                                "value": object_name,
                            }
                        },
                        "assign_to": "obj_pose",
                    },
                    {
                        "kind": "feedback",
                        "description": (
                            f"Is {object_name} visible and reachable?"
                        ),
                    },
                    {
                        "kind": "call",
                        "skill": "arm.grab",
                        "args": {
                            "object_name": {
                                "kind": "literal",
                                "value": object_name,
                            }
                        },
                        "assign_to": "grab_result",
                    },
                    {
                        "kind": "feedback",
                        "description": f"Is {object_name} grasped?",
                    },
                    {
                        "kind": "return",
                        "value": {"kind": "var", "name": "grab_result"},
                    },
                ],
            },
        }
    )
