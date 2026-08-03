"""Plan AST validation tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from plan import (
    BinOpExpr,
    CallNode,
    Plan,
    VarExpr,
    WhileNode,
    get_stop_reason,
    is_a_stop_plan,
    validate_plan,
)


def test_valid_plan_roundtrip() -> None:
    data = {
        "goal": "demo goal",
        "body": {
            "kind": "seq",
            "steps": [
                {
                    "kind": "if",
                    "condition": {"kind": "literal", "value": True},
                    "then_branch": {
                        "kind": "call",
                        "skill": "arm.grab",
                        "args": {"object_name": {"kind": "literal", "value": "cube"}},
                    },
                    "else_branch": {"kind": "feedback", "description": "check branch"},
                },
                {
                    "kind": "while",
                    "condition": {"kind": "literal", "value": False},
                    "body": {"kind": "feedback", "description": "loop check"},
                },
                {"kind": "return", "value": {"kind": "literal", "value": "done"}},
            ],
        },
    }
    plan = validate_plan(data)
    assert isinstance(plan, Plan)
    dumped = plan.model_dump()
    assert dumped["goal"] == data["goal"]
    assert dumped["body"]["kind"] == "seq"
    assert dumped["body"]["steps"][0]["kind"] == "if"
    assert dumped["body"]["steps"][1]["kind"] == "while"
    assert dumped["body"]["steps"][2]["kind"] == "return"


def test_reject_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        validate_plan({"goal": "...", "body": {"kind": "unknown"}})


def test_reject_missing_goal() -> None:
    with pytest.raises(ValidationError):
        validate_plan({"body": {"kind": "return"}})


def test_expr_discriminator() -> None:
    expr = BinOpExpr.model_validate(
        {
            "kind": "binop",
            "op": "==",
            "left": {"kind": "var", "name": "x"},
            "right": {"kind": "literal", "value": 1},
        }
    )
    assert expr.kind == "binop"
    assert isinstance(expr.left, VarExpr)
    assert expr.left.name == "x"
    assert expr.right.kind == "literal"


def test_call_assign_to_optional() -> None:
    node = CallNode.model_validate({"kind": "call", "skill": "arm.grab"})
    assert node.assign_to is None


def test_call_args_coerce_bare_primitives() -> None:
    data = {
        "goal": "place objects",
        "body": {
            "kind": "call",
            "skill": "arm.place",
            "args": {"object_name": "cup", "count": 2},
        },
    }
    plan = validate_plan(data)
    call = plan.body
    assert isinstance(call, CallNode)
    assert call.args["object_name"].kind == "literal"
    assert call.args["object_name"].value == "cup"
    assert call.args["count"].kind == "literal"
    assert call.args["count"].value == 2


def test_binop_in_valid() -> None:
    expr = BinOpExpr.model_validate(
        {
            "kind": "binop",
            "op": "in",
            "left": {"kind": "literal", "value": "apple"},
            "right": {"kind": "var", "name": "objects"},
        }
    )
    assert expr.op == "in"


def test_binop_bad_op_rejected() -> None:
    with pytest.raises(ValidationError):
        BinOpExpr.model_validate(
            {
                "kind": "binop",
                "op": "not_an_op",
                "left": {"kind": "literal", "value": 1},
                "right": {"kind": "literal", "value": 2},
            }
        )


def test_while_default_max_iterations() -> None:
    node = WhileNode.model_validate(
        {
            "kind": "while",
            "condition": {"kind": "literal", "value": True},
            "body": {"kind": "feedback", "description": "still running"},
        }
    )
    assert node.max_iterations == 20


def test_reject_bare_string_in_steps() -> None:
    with pytest.raises(ValidationError):
        validate_plan(
            {
                "goal": "bad",
                "body": {"kind": "seq", "steps": ["else_branch"]},
            }
        )


def test_stop_plan_detects_sentinel() -> None:
    data = {
        "goal": "g",
        "body": {
            "kind": "return",
            "value": {"kind": "literal", "value": "STOP: goal achieved"},
        },
    }
    assert is_a_stop_plan(data)
    assert get_stop_reason(data) == "goal achieved"


def test_stop_plan_rejects_normal_plan() -> None:
    data = {
        "goal": "g",
        "body": {
            "kind": "return",
            "value": {"kind": "literal", "value": "done"},
        },
    }
    assert not is_a_stop_plan(data)


def test_stop_plan_rejects_stop_json() -> None:
    assert not is_a_stop_plan({"stop": True, "reason": "nope"})
