"""Plan AST models and validation for LLM-emitted robot task plans."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class LiteralExpr(BaseModel):
    """Literal value expression."""

    kind: Literal["literal"] = "literal"
    value: bool | int | float | str | None


class VarExpr(BaseModel):
    """Variable reference expression."""

    kind: Literal["var"] = "var"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value or not value.isidentifier():
            raise ValueError("name must be a non-empty identifier")
        return value


class BinOpExpr(BaseModel):
    """Binary operator expression."""

    kind: Literal["binop"] = "binop"
    op: Literal["==", "!=", "<", "<=", ">", ">=", "and", "or", "in"]
    left: "Expr"
    right: "Expr"


class NotExpr(BaseModel):
    """Logical negation expression."""

    kind: Literal["not"] = "not"
    operand: "Expr"


Expr = Annotated[
    Union[LiteralExpr, VarExpr, BinOpExpr, NotExpr],
    Field(discriminator="kind"),
]


class SeqNode(BaseModel):
    """Sequential statement list."""

    kind: Literal["seq"] = "seq"
    steps: list["StmtNode"]


class IfNode(BaseModel):
    """Conditional branch statement."""

    kind: Literal["if"] = "if"
    condition: Expr
    then_branch: "StmtNode"
    else_branch: "StmtNode | None" = None


class WhileNode(BaseModel):
    """Bounded while-loop statement."""

    kind: Literal["while"] = "while"
    condition: Expr
    body: "StmtNode"
    max_iterations: int = 20


class CallNode(BaseModel):
    """Skill invocation statement."""

    kind: Literal["call"] = "call"
    skill: str
    args: dict[str, Expr] = Field(default_factory=dict)
    assign_to: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_args(cls, data: object) -> object:
        if isinstance(data, dict):
            args = data.get("args")
            if isinstance(args, dict):
                data["args"] = {
                    k: (
                        {"kind": "literal", "value": v}
                        if isinstance(v, (str, int, float, bool))
                        else v
                    )
                    for k, v in args.items()
                }
        return data


class FeedbackNode(BaseModel):
    """Planner feedback checkpoint statement."""

    kind: Literal["feedback"] = "feedback"
    description: str


class ReturnNode(BaseModel):
    """Early return from plan execution."""

    kind: Literal["return"] = "return"
    value: Expr | None = None


StmtNode = Annotated[
    Union[SeqNode, IfNode, WhileNode, CallNode, FeedbackNode, ReturnNode],
    Field(discriminator="kind"),
]


class Plan(BaseModel):
    """Validated plan AST root (goal plus body)."""

    goal: str
    body: StmtNode


BinOpExpr.model_rebuild()
NotExpr.model_rebuild()
SeqNode.model_rebuild()
IfNode.model_rebuild()
WhileNode.model_rebuild()
CallNode.model_rebuild()
ReturnNode.model_rebuild()
Plan.model_rebuild()


def validate_plan(data: dict) -> Plan:
    """Parse and validate a plan dict into a ``Plan``."""
    return Plan.model_validate(data)


def _stop_reason(data: Plan | dict) -> str | None:
    """Return stop reason if data is a sentinel STOP plan, else None."""
    if isinstance(data, Plan):
        body = data.body
        if not isinstance(body, ReturnNode):
            return None
        if not isinstance(body.value, LiteralExpr):
            return None
        raw = body.value.value
        if not isinstance(raw, str) or not raw.startswith("STOP:"):
            return None
        reason = raw[5:].strip()
        return reason if reason else "planner stopped"

    body = data.get("body")
    if not isinstance(body, dict) or body.get("kind") != "return":
        return None
    value = body.get("value")
    if not isinstance(value, dict) or value.get("kind") != "literal":
        return None
    raw = value.get("value")
    if not isinstance(raw, str) or not raw.startswith("STOP:"):
        return None
    reason = raw[5:].strip()
    return reason if reason else "planner stopped"


def is_a_stop_plan(data: Plan | dict) -> bool:
    """Return True if data is a sentinel STOP plan."""
    return _stop_reason(data) is not None


def get_stop_reason(data: Plan | dict) -> str:
    """Return the stop reason from a sentinel STOP plan.

    Raises:
        ValueError: If ``data`` is not a STOP plan.
    """
    reason = _stop_reason(data)
    if reason is None:
        raise ValueError("plan is not a STOP plan")
    return reason
