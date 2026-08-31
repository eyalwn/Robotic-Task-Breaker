"""Manual live-LLM planner test using a real LIBERO start snapshot."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from plan import Plan, is_stop_plan
from planner import LLMPlanner
from skills import skill_catalog

SNAPSHOT_PATH = (
    Path(__file__).parent / "fixtures" / "libero_object_task0_start.jpg"
)
NUM_RUNS = 3
GOAL = "Pick the alphabet soup and place it in the basket"

ALLOWED_SKILLS = {"grab", "place", "close", "turn_on"}

CATALOG = skill_catalog()


def _plan_json(plan: Plan) -> str:
    return json.dumps(plan, indent=2)


def _assert_plan_semantics(plan: Plan) -> None:
    plan_json = _plan_json(plan)
    assert not is_stop_plan(plan), plan_json
    assert plan, plan_json
    for step in plan:
        assert step["skill"] in ALLOWED_SKILLS, (
            f"unknown skill {step['skill']!r} in plan:\n{plan_json}"
        )


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
