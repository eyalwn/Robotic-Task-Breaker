"""Restricted expression evaluator tests."""
from __future__ import annotations

import pytest

from conftest import MockSkill, build_executor, make_plan, run_plan
from plan import BinOpExpr, LiteralExpr, NotExpr, VarExpr


def _return_value(expr_dict: dict) -> object:
    plan = make_plan({"kind": "return", "value": expr_dict})
    return run_plan(build_executor(MockSkill()), plan).return_value


def test_literal() -> None:
    assert _return_value({"kind": "literal", "value": 7}) == 7
    assert _return_value({"kind": "literal", "value": "hi"}) == "hi"
    assert _return_value({"kind": "literal", "value": None}) is None


def test_var_bound() -> None:
    ex = build_executor(MockSkill())
    assert ex._evaluate_expr(VarExpr(name="x"), {"x": 99}) == 99


def test_var_unbound_raises() -> None:
    ex = build_executor(MockSkill())
    with pytest.raises(NameError, match="not bound"):
        ex._evaluate_expr(VarExpr(name="missing"), {})


def test_binop_equality() -> None:
    assert (
        _return_value(
            {
                "kind": "binop",
                "op": "==",
                "left": {"kind": "literal", "value": 1},
                "right": {"kind": "literal", "value": 1},
            }
        )
        is True
    )
    assert (
        _return_value(
            {
                "kind": "binop",
                "op": "==",
                "left": {"kind": "literal", "value": 1},
                "right": {"kind": "literal", "value": 2},
            }
        )
        is False
    )


def test_strict_equality_dict_vs_string() -> None:
    ex = build_executor(MockSkill())
    result = ex._evaluate_expr(
        BinOpExpr(
            op="==",
            left=VarExpr(name="outcome"),
            right=LiteralExpr(value="success"),
        ),
        {"outcome": {"status": "success"}},
    )
    assert result is False


def test_strict_equality_json_string_vs_string() -> None:
    value = _return_value(
        {
            "kind": "binop",
            "op": "==",
            "left": {"kind": "literal", "value": '{"status":"success"}'},
            "right": {"kind": "literal", "value": "success"},
        }
    )
    assert value is False


def test_strict_inequality_same_type() -> None:
    assert (
        _return_value(
            {
                "kind": "binop",
                "op": "!=",
                "left": {"kind": "literal", "value": "failed"},
                "right": {"kind": "literal", "value": "success"},
            }
        )
        is True
    )


def test_binop_and_short_circuit() -> None:
    ex = build_executor(MockSkill())
    result = ex._evaluate_expr(
        BinOpExpr(
            op="and",
            left=LiteralExpr(value=False),
            right=VarExpr(name="missing"),
        ),
        {},
    )
    assert result is False


def test_binop_or_short_circuit() -> None:
    ex = build_executor(MockSkill())
    result = ex._evaluate_expr(
        BinOpExpr(
            op="or",
            left=LiteralExpr(value=True),
            right=VarExpr(name="missing"),
        ),
        {},
    )
    assert result is True


def test_binop_in_true() -> None:
    ex = build_executor(MockSkill())
    result = ex._evaluate_expr(
        BinOpExpr(
            op="in",
            left=LiteralExpr(value="apple"),
            right=VarExpr(name="objects"),
        ),
        {"objects": ["apple", "banana"]},
    )
    assert result is True


def test_binop_in_none_right() -> None:
    assert (
        _return_value(
            {
                "kind": "binop",
                "op": "in",
                "left": {"kind": "literal", "value": "apple"},
                "right": {"kind": "literal", "value": None},
            }
        )
        is False
    )


def test_not_expr() -> None:
    ex = build_executor(MockSkill())
    result = ex._evaluate_expr(NotExpr(operand=LiteralExpr(value=True)), {})
    assert result is False


def test_incomparable_operands_type_error() -> None:
    ex = build_executor(MockSkill())
    with pytest.raises(TypeError, match="Incomparable operands"):
        ex._evaluate_expr(
            BinOpExpr(
                op="<",
                left=LiteralExpr(value="a"),
                right=LiteralExpr(value=1),
            ),
            {},
        )
