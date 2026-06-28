from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class LiteralExpr(BaseModel):
    kind: Literal["literal"] = "literal"
    value: bool | int | float | str | None


class VarExpr(BaseModel):
    kind: Literal["var"] = "var"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value or not value.isidentifier():
            raise ValueError("name must be a non-empty identifier")
        return value


class BinOpExpr(BaseModel):
    kind: Literal["binop"] = "binop"
    op: Literal["==", "!=", "<", "<=", ">", ">=", "and", "or", "in"]
    left: "Expr"
    right: "Expr"


class NotExpr(BaseModel):
    kind: Literal["not"] = "not"
    operand: "Expr"


Expr = Annotated[
    Union[LiteralExpr, VarExpr, BinOpExpr, NotExpr],
    Field(discriminator="kind"),
]


class SeqNode(BaseModel):
    kind: Literal["seq"] = "seq"
    steps: list["StmtNode"]


class IfNode(BaseModel):
    kind: Literal["if"] = "if"
    condition: Expr
    then_branch: "StmtNode"
    else_branch: "StmtNode | None" = None


class WhileNode(BaseModel):
    kind: Literal["while"] = "while"
    condition: Expr
    body: "StmtNode"
    max_iterations: int = 20


class CallNode(BaseModel):
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
    kind: Literal["feedback"] = "feedback"
    description: str


class ReturnNode(BaseModel):
    kind: Literal["return"] = "return"
    value: Expr | None = None


StmtNode = Annotated[
    Union[SeqNode, IfNode, WhileNode, CallNode, FeedbackNode, ReturnNode],
    Field(discriminator="kind"),
]


class Plan(BaseModel):
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
    return Plan.model_validate(data)
