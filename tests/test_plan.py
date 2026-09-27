"""Linear plan helper tests."""
from __future__ import annotations

import pytest

from plan import get_stop_reason, is_stop_plan, parse_planner_payload


def test_parse_step_array() -> None:
    data = [
        {
            "skill": "transfer",
            "object_name": "alphabet soup",
            "target_name": "basket",
        },
    ]
    plan = parse_planner_payload(data)
    assert plan == data
    assert plan[0]["skill"] == "transfer"
    assert plan[0]["target_name"] == "basket"


def test_reject_wrapped_steps() -> None:
    with pytest.raises(ValueError, match="plan must be"):
        parse_planner_payload(
            {"steps": [{"skill": "transfer", "object_name": "alphabet soup"}]}
        )


def test_reject_missing_steps_and_stop() -> None:
    with pytest.raises(ValueError, match="plan must be"):
        parse_planner_payload({})


def test_reject_invalid_step() -> None:
    with pytest.raises(ValueError, match="each step requires"):
        parse_planner_payload([{"object_name": "x"}])


def test_parse_open_drawer_step() -> None:
    plan = parse_planner_payload([{"skill": "open_drawer"}])
    assert plan == [{"skill": "open_drawer"}]


def test_reject_open_drawer_with_transfer_fields() -> None:
    with pytest.raises(ValueError, match="must not include transfer fields"):
        parse_planner_payload([{"skill": "open_drawer", "object_name": "drawer"}])


def test_reject_non_transfer_skill() -> None:
    with pytest.raises(ValueError, match="unsupported skill"):
        parse_planner_payload([{"skill": "grab", "object_name": "x"}])


def test_reject_transfer_missing_target_name() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'target_name'"):
        parse_planner_payload([{"skill": "transfer", "object_name": "x"}])


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
            "skill": "transfer",
            "object_name": "x",
            "target_name": "basket",
        }
    ]
    assert not is_stop_plan(parse_planner_payload(data))


def test_stop_plan_rejects_legacy_json() -> None:
    assert not is_stop_plan({"stop": True, "reason": "nope"})
