"""Linear plan helper tests."""
from __future__ import annotations

import pytest

from plan import get_stop_reason, is_stop_plan, parse_planner_payload

OPEN_STEP = {"skill": "open", "object_name": "right drawer"}
PLACE_STEP = {
    "skill": "place",
    "object_name": "kettle",
    "source_name": "counter",
    "destination_name": "rear right burner of the stove",
}
TURN_ON_STEP = {
    "skill": "turn_on",
    "target_name": "rear right burner of the stove",
}
PUT_IN_STEP = {
    "skill": "put_in",
    "object_name": "bar",
    "source_name": "drawer",
    "container_name": "bowl",
}
CLOSE_STEP = {"skill": "close", "object_name": "drawer"}


def test_parse_open_step_array() -> None:
    plan = parse_planner_payload([OPEN_STEP])
    assert plan == [OPEN_STEP]


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
            [{"skill": "open", "object_name": "drawer", "source_name": "x"}]
        )


def test_parse_place_step() -> None:
    plan = parse_planner_payload([PLACE_STEP])
    assert plan == [PLACE_STEP]


def test_reject_place_missing_arg() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'destination_name'"):
        parse_planner_payload(
            [{"skill": "place", "object_name": "k", "source_name": "c"}]
        )


def test_reject_place_blank_arg() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'source_name'"):
        parse_planner_payload(
            [{
                "skill": "place",
                "object_name": "k",
                "source_name": "",
                "destination_name": "burner",
            }]
        )


def test_reject_place_extra_key() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_planner_payload([{**PLACE_STEP, "extra": "x"}])


def test_parse_turn_on_step() -> None:
    plan = parse_planner_payload([TURN_ON_STEP])
    assert plan == [TURN_ON_STEP]


def test_parse_put_in_step() -> None:
    plan = parse_planner_payload([PUT_IN_STEP])
    assert plan == [PUT_IN_STEP]


def test_reject_put_in_missing_arg() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'container_name'"):
        parse_planner_payload(
            [{
                "skill": "put_in",
                "object_name": "bar",
                "source_name": "drawer",
            }]
        )


def test_reject_put_in_extra_key() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_planner_payload([{**PUT_IN_STEP, "destination_name": "bowl"}])


def test_parse_close_step() -> None:
    plan = parse_planner_payload([CLOSE_STEP])
    assert plan == [CLOSE_STEP]


def test_reject_close_missing_object_name() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'object_name'"):
        parse_planner_payload([{"skill": "close"}])


def test_reject_close_extra_key() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_planner_payload(
            [{"skill": "close", "object_name": "drawer", "source_name": "x"}]
        )


def test_reject_turn_on_missing_target_name() -> None:
    with pytest.raises(ValueError, match="requires non-empty 'target_name'"):
        parse_planner_payload([{"skill": "turn_on"}])


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
