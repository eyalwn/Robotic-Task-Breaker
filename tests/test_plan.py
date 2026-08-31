"""Linear plan helper tests."""
from __future__ import annotations

import pytest

from plan import get_stop_reason, is_stop_plan, parse_planner_payload


def test_parse_step_array() -> None:
    data = [
        {"skill": "grab", "object_name": "alphabet soup"},
        {
            "skill": "place",
            "object_name": "alphabet soup",
            "target_name": "basket",
            "position": "inside",
        },
        {"skill": "close", "target_name": "drawer"},
    ]
    plan = parse_planner_payload(data)
    assert plan == data
    assert plan[0]["skill"] == "grab"
    assert plan[1]["position"] == "inside"


def test_reject_wrapped_steps() -> None:
    with pytest.raises(ValueError, match="plan must be"):
        parse_planner_payload(
            {"steps": [{"skill": "grab", "object_name": "alphabet soup"}]}
        )


def test_reject_missing_steps_and_stop() -> None:
    with pytest.raises(ValueError, match="plan must be"):
        parse_planner_payload({})


def test_reject_invalid_step() -> None:
    with pytest.raises(ValueError, match="each step requires"):
        parse_planner_payload([{"object_name": "x"}])


def test_empty_steps_allowed() -> None:
    plan = parse_planner_payload([])
    assert plan == []


def test_stop_plan_detects_sentinel() -> None:
    data = {"stop": "goal achieved"}
    assert is_stop_plan(parse_planner_payload(data))
    assert get_stop_reason(parse_planner_payload(data)) == "goal achieved"


def test_stop_plan_rejects_normal_plan() -> None:
    data = [
        {
            "skill": "grab",
            "object_name": "x",
        }
    ]
    assert not is_stop_plan(parse_planner_payload(data))


def test_stop_plan_rejects_legacy_json() -> None:
    assert not is_stop_plan({"stop": True, "reason": "nope"})
