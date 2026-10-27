"""Linear plan helper tests."""
from __future__ import annotations

import pytest

from plan import get_stop_reason, is_stop_plan, parse_planner_payload

OPEN_STEP = {"skill": "open", "object_name": "right drawer"}
CLOSE_STEP = {"skill": "close", "object_name": "drawer"}
OPEN_CLOSE_PLAN = [{"skill": "open", "object_name": "left drawer"},
                   {"skill": "close", "object_name": "left drawer"}]


def test_parse_open_step_array() -> None:
    plan = parse_planner_payload([OPEN_STEP])
    assert plan == [OPEN_STEP]


def test_parse_open_close_chain() -> None:
    plan = parse_planner_payload(OPEN_CLOSE_PLAN)
    assert plan == OPEN_CLOSE_PLAN


def test_reject_wrapped_steps() -> None:
    with pytest.raises(ValueError, match="plan must be"):
        parse_planner_payload({"steps": [OPEN_STEP]})


def test_reject_missing_steps_and_stop() -> None:
    with pytest.raises(ValueError, match="plan must be"):
        parse_planner_payload({})


def test_reject_invalid_step() -> None:
    with pytest.raises(ValueError, match="each step requires"):
        parse_planner_payload([{"object_name": "x"}])


def test_reject_open_missing_object_name() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'object_name'"):
        parse_planner_payload([{"skill": "open"}])


def test_reject_open_legacy_target_name() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_planner_payload([{"skill": "open", "target_name": "drawer"}])


def test_reject_open_blank_object_name() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'object_name'"):
        parse_planner_payload([{"skill": "open", "object_name": "   "}])


def test_reject_open_extra_key() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_planner_payload(
            [{"skill": "open", "object_name": "drawer", "source_name": "x"}])


def test_parse_close_step() -> None:
    plan = parse_planner_payload([CLOSE_STEP])
    assert plan == [CLOSE_STEP]


def test_reject_close_missing_object_name() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'object_name'"):
        parse_planner_payload([{"skill": "close"}])


def test_reject_close_extra_key() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_planner_payload(
            [{"skill": "close", "object_name": "drawer", "source_name": "x"}])


def test_reject_removed_place_skill() -> None:
    with pytest.raises(ValueError, match="unsupported skill"):
        parse_planner_payload([{"skill": "place", "object_name": "k",
                                "source_name": "c", "destination_name": "d"}])


def test_reject_removed_put_in_skill() -> None:
    with pytest.raises(ValueError, match="unsupported skill"):
        parse_planner_payload([{"skill": "put_in", "object_name": "bar",
                                "source_name": "drawer", "container_name": "bowl"}])


def test_reject_removed_turn_on_skill() -> None:
    with pytest.raises(ValueError, match="unsupported skill"):
        parse_planner_payload([{"skill": "turn_on", "target_name": "burner"}])


def test_reject_legacy_open_drawer_skill() -> None:
    with pytest.raises(ValueError, match="unsupported skill"):
        parse_planner_payload([{"skill": "open_drawer"}])


def test_reject_unsupported_skill() -> None:
    with pytest.raises(ValueError, match="unsupported skill"):
        parse_planner_payload([{"skill": "grab", "object_name": "x"}])


def test_empty_steps_allowed() -> None:
    plan = parse_planner_payload([])
    assert plan == []


def test_stop_plan_detects_sentinel() -> None:
    data = {"stop": "goal achieved"}
    assert is_stop_plan(parse_planner_payload(data))
    assert get_stop_reason(parse_planner_payload(data)) == "goal achieved"


def test_stop_plan_rejects_normal_plan() -> None:
    assert not is_stop_plan(parse_planner_payload([OPEN_STEP]))


def test_stop_plan_rejects_legacy_json() -> None:
    assert not is_stop_plan({"stop": True, "reason": "nope"})
