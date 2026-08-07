"""Manual live-LLM planner test using a real LIBERO start snapshot."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from plan import (
    CallNode,
    IfNode,
    Plan,
    SeqNode,
    StmtNode,
    WhileNode,
    is_a_stop_plan,
)
from planner import LLMPlanner
from skills import arm_skill_catalog

SNAPSHOT_PATH = (
    Path(__file__).parent / "fixtures" / "libero_object_task0_start.jpg"
)
NUM_RUNS = 3
GOAL = "Pick the alphabet soup and place it in the basket"

ALLOWED_SKILLS = {
    "arm.grab",
    "arm.place",
    "arm.transfer",
    "arm.open",
    "arm.close",
    "arm.push",
    "arm.pull",
}

CATALOG = arm_skill_catalog()


def _iter_calls(stmt: StmtNode):
    if isinstance(stmt, CallNode):
        yield stmt
        return
    if isinstance(stmt, SeqNode):
        for step in stmt.steps:
            yield from _iter_calls(step)
    elif isinstance(stmt, IfNode):
        yield from _iter_calls(stmt.then_branch)
        if stmt.else_branch is not None:
            yield from _iter_calls(stmt.else_branch)
    elif isinstance(stmt, WhileNode):
        yield from _iter_calls(stmt.body)


def _plan_json(plan: Plan) -> str:
    return json.dumps(plan.model_dump(), indent=2)


def _assert_plan_semantics(plan: Plan) -> None:
    plan_json = _plan_json(plan)
    assert not is_a_stop_plan(plan), plan_json

    calls = list(_iter_calls(plan.body))
    assert calls, plan_json

    skills = [call.skill for call in calls]
    for skill in skills:
        assert skill in ALLOWED_SKILLS, f"unknown skill {skill!r} in plan:\n{plan_json}"


@pytest.mark.integration
def test_planner_live_llm_initial_plan(
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    if (pytestconfig.getoption("-m") or "").strip() != "integration":
        pytest.skip(
            "manual live test; run with: pytest -m integration "
            "tests/test_planner_live_llm.py -q"
        )
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if not SNAPSHOT_PATH.exists():
        pytest.skip(
            "real snapshot missing: download one and place it at "
            f"{SNAPSHOT_PATH.as_posix()}"
        )

    snapshot_bytes = SNAPSHOT_PATH.read_bytes()
    monkeypatch.setattr("planner.config.MAX_PLAN_RETRIES", 1)
    for _ in range(NUM_RUNS):
        planner = LLMPlanner(GOAL, CATALOG)
        plan = planner.plan(snapshot_bytes)
        _assert_plan_semantics(plan)
