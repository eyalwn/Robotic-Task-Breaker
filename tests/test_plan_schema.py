from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.plan import BinOpExpr, CallNode, Plan, VarExpr, WhileNode, validate_plan


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
    assert expr.left.kind == "var"
    assert expr.right.kind == "literal"


def test_call_assign_to_optional() -> None:
    node = CallNode.model_validate({"kind": "call", "skill": "arm.grab"})
    assert node.assign_to is None


def test_while_default_max_iterations() -> None:
    node = WhileNode.model_validate(
        {
            "kind": "while",
            "condition": {"kind": "literal", "value": True},
            "body": {"kind": "feedback", "description": "still running"},
        }
    )
    assert node.max_iterations == 20
