"""Plan helpers: step lists and STOP sentinel detection."""
from __future__ import annotations

from typing import TypeAlias

Step: TypeAlias = dict[str, str]
Plan: TypeAlias = list[Step]

_SKILL_ARG_KEYS: dict[str, tuple[str, ...]] = {
    "open": ("object_name",),
    "close": ("object_name",),
}


def is_stop_plan(data: object) -> bool:
    """Return True if data is a sentinel STOP plan."""
    if not isinstance(data, dict):
        return False
    stop = data.get("stop")
    return isinstance(stop, str) and bool(stop.strip())


def get_stop_reason(data: object) -> str:
    """Return the stop reason from a sentinel STOP plan."""
    if not is_stop_plan(data):
        raise ValueError("plan is not a STOP plan")
    reason = str(data["stop"]).strip()
    return reason if reason else "planner stopped"


def parse_planner_payload(data: object) -> object:
    """Parse validated planner JSON into steps or a STOP sentinel."""
    if isinstance(data, list):
        return [_parse_step(step) for step in data]
    if not isinstance(data, dict):
        raise ValueError("plan must be a JSON array or object")
    stop = data.get("stop")
    if isinstance(stop, str) and stop.strip():
        return {"stop": stop.strip()}
    raise ValueError("plan must be a step array or include 'stop'")


def _parse_step(step: object) -> Step:
    """Validate a flat step dict with string values."""
    if not isinstance(step, dict):
        raise ValueError("each step must be an object")
    skill = step.get("skill")
    if not isinstance(skill, str) or not skill.strip():
        raise ValueError("each step requires a non-empty 'skill'")
    skill_name = skill.strip()
    arg_keys = _SKILL_ARG_KEYS.get(skill_name)
    if arg_keys is None:
        raise ValueError(f"unsupported skill: {skill_name!r}")
    allowed = frozenset({"skill", *arg_keys})
    extra = set(step.keys()) - allowed
    if extra:
        names = ", ".join(sorted(str(k) for k in extra))
        raise ValueError(f"{skill_name} step has unexpected keys: {names}")
    out: Step = {"skill": skill_name}
    for key in arg_keys:
        value = step.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{skill_name} step requires non-empty {key!r}")
        out[key] = value.strip()
    return out
