"""Unit tests for no-LLM plan stubs (StaticPlanner, AlwaysContinueEvaluator)."""
from core.plan import StaticPlanner, validate_plan
from core.plan.feedback import AlwaysContinueEvaluator, FeedbackDecision


def test_static_planner_returns_preset_plan():
    plan = validate_plan(
        {
            "goal": "pick the cube",
            "body": {"kind": "return", "value": {"kind": "literal", "value": True}},
        }
    )
    planner = StaticPlanner(plan)
    assert planner.plan("g", "cat") is plan
    assert planner.plan("other", "catalog", context="ctx") is plan


def test_always_continue_evaluator():
    outcome = AlwaysContinueEvaluator().evaluate(b"", "d", "g")
    assert outcome.decision == FeedbackDecision.CONTINUE
