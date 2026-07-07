from .executor import (
    ExecutionResult,
    Executor,
    ReplanSignal,
    ReturnSignal,
    StepBudgetExceeded,
    StopSignal,
)
from .feedback import (
    AlwaysContinueEvaluator,
    FeedbackDecision,
    FeedbackEvaluator,
    FeedbackOutcome,
)
from .planner import Planner, StaticPlanner
from .schema import (
    BinOpExpr,
    CallNode,
    Expr,
    FeedbackNode,
    IfNode,
    LiteralExpr,
    NotExpr,
    Plan,
    ReturnNode,
    SeqNode,
    StmtNode,
    VarExpr,
    WhileNode,
    validate_plan,
)
from .trace import TraceWriter

__all__ = [
    "Plan",
    "validate_plan",
    "Expr",
    "StmtNode",
    "LiteralExpr",
    "VarExpr",
    "BinOpExpr",
    "NotExpr",
    "SeqNode",
    "IfNode",
    "WhileNode",
    "CallNode",
    "FeedbackNode",
    "ReturnNode",
    "FeedbackDecision",
    "FeedbackOutcome",
    "FeedbackEvaluator",
    "AlwaysContinueEvaluator",
    "Planner",
    "StaticPlanner",
    "LLMPlanner",
    "LLMFeedbackEvaluator",
    "Executor",
    "ReplanSignal",
    "ReturnSignal",
    "StopSignal",
    "StepBudgetExceeded",
    "ExecutionResult",
    "TraceWriter",
]


def __getattr__(name: str):
    if name == "LLMPlanner":
        from .llm_planner import LLMPlanner

        return LLMPlanner
    if name == "LLMFeedbackEvaluator":
        from .llm_feedback import LLMFeedbackEvaluator

        return LLMFeedbackEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
