"""LLM planner tests with mocked chat model (no OpenAI)."""
from __future__ import annotations

import json

import pytest

from plan import Plan
from planner import (
    AlwaysContinueEvaluator,
    FeedbackDecision,
    FeedbackHandoff,
    LLMPlanner,
    PlannerAction,
    _extract_json_object,
    _strip_code_fences,
)

VALID_PLAN = {
    "goal": "pick the cube",
    "body": {
        "kind": "call",
        "skill": "arm.grab",
        "args": {"object_name": {"kind": "literal", "value": "cube"}},
    },
}


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = None


class _LLM:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def bind(self, **_kwargs: object) -> _LLM:
        return self

    def bind_tools(self, _tools: list, tool_choice: str | None = None) -> _LLM:
        return self

    def invoke(self, _messages: list) -> _Response:
        if self._idx >= len(self._responses):
            return self._responses[-1]
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class _SpyLLM(_LLM):
    def __init__(self, responses: list[_Response]) -> None:
        super().__init__(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list) -> _Response:
        self.calls.append(messages)
        return super().invoke(messages)


class _StubFeedbackEvaluator:
    def __init__(self, decision: FeedbackDecision, reason: str = "") -> None:
        self.decision = decision
        self.reason = reason
        self.last_description: str | None = None

    def evaluate(self, image_bytes: bytes, description: str, goal: str):
        self.last_description = description
        from planner import FeedbackOutcome

        return FeedbackOutcome(decision=self.decision, reason=self.reason)


def _mock_llm(plan_dict: dict) -> _LLM:
    return _LLM([_Response(json.dumps(plan_dict))])


def _patch_planner_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "planner.LLMFeedbackEvaluator",
        lambda model: AlwaysContinueEvaluator(),
    )


def test_strip_code_fences() -> None:
    raw = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    stripped = _strip_code_fences(raw)
    assert stripped.startswith("{")
    assert stripped.endswith("}")


def test_extract_json_object_from_fenced_output() -> None:
    raw = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    parsed = json.loads(_extract_json_object(raw))
    assert parsed["goal"] == "pick the cube"


def test_extract_json_object_with_trailing_prose() -> None:
    raw = f"{json.dumps(VALID_PLAN)}\n\nNote: use feedback after grabs."
    parsed = json.loads(_extract_json_object(raw))
    assert parsed["goal"] == "pick the cube"


def test_extract_json_object_brace_balanced_with_string_braces() -> None:
    raw = (
        'prefix {"goal": "g", "body": {"kind": "literal", "value": "has } brace"}} '
        "suffix"
    )
    parsed = json.loads(_extract_json_object(raw))
    assert parsed["goal"] == "g"
    assert parsed["body"]["value"] == "has } brace"


def test_planner_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner_evaluator(monkeypatch)
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner("pick cube", "catalog", [])
    plan = planner.plan()
    assert isinstance(plan, Plan)
    assert plan.goal == "pick the cube"


def test_planner_retry_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner_evaluator(monkeypatch)
    responses = [
        _Response("{bad json"),
        _Response(json.dumps(VALID_PLAN)),
    ]
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM(responses))
    monkeypatch.setattr("planner.config.MAX_PLAN_RETRIES", 2)
    planner = LLMPlanner("pick cube", "catalog", [])
    plan = planner.plan()
    assert isinstance(plan, Plan)


def test_planner_retry_prompt_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner_evaluator(monkeypatch)
    invalid = {
        "goal": "pick the cube",
        "body": {"kind": "seq", "steps": ["else_branch"]},
    }
    spy = _SpyLLM(
        [
            _Response(json.dumps(invalid)),
            _Response(json.dumps(VALID_PLAN)),
        ]
    )
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    monkeypatch.setattr("planner.config.MAX_PLAN_RETRIES", 2)
    planner = LLMPlanner("pick cube", "catalog", [])
    plan = planner.plan()
    assert isinstance(plan, Plan)
    assert len(spy.calls) == 2
    retry_msg = spy.calls[1][-1].content
    assert "never a bare string like 'else_branch'" in retry_msg


def test_planner_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner_evaluator(monkeypatch)
    responses = [_Response("{bad"), _Response("{bad")]
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM(responses))
    monkeypatch.setattr("planner.config.MAX_PLAN_RETRIES", 1)
    planner = LLMPlanner("pick cube", "catalog", [])
    with pytest.raises(RuntimeError, match="plan validation failed"):
        planner.plan()


def test_planner_strips_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner_evaluator(monkeypatch)
    fenced = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    monkeypatch.setattr(
        "planner.init_chat_model",
        lambda _model: _LLM([_Response(fenced)]),
    )
    planner = LLMPlanner("pick cube", "catalog", [])
    plan = planner.plan()
    assert isinstance(plan, Plan)


def test_planner_shape_goal_with_scene_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner_evaluator(monkeypatch)
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner(
        "put items in basket",
        "catalog",
        ["alphabet_soup_1", "cream_cheese_1"],
    )
    planner.plan()
    human_msg = spy.calls[0][-1].content
    assert "put items in basket" in human_msg
    assert "alphabet_soup_1" in human_msg
    assert "Available scene objects" in human_msg


def test_planner_shape_goal_without_scene_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner_evaluator(monkeypatch)
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("pick cube", "catalog", [])
    planner.plan()
    human_msg = spy.calls[0][-1].content
    assert human_msg == "Goal: pick cube"


def test_planner_decide_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubFeedbackEvaluator(FeedbackDecision.CONTINUE, "ok")
    monkeypatch.setattr("planner.LLMFeedbackEvaluator", lambda model: stub)
    planner = LLMPlanner("goal", "catalog", [])
    handoff = FeedbackHandoff(
        description="check grasp",
        prompt="check grasp",
        image_bytes=b"img",
    )
    decision = planner.decide(handoff)
    assert decision.action == PlannerAction.CONTINUE
    assert decision.reason == "ok"


def test_planner_decide_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubFeedbackEvaluator(FeedbackDecision.STOP, "done")
    monkeypatch.setattr("planner.LLMFeedbackEvaluator", lambda model: stub)
    planner = LLMPlanner("goal", "catalog", [])
    handoff = FeedbackHandoff(
        description="all placed",
        prompt="all placed",
        image_bytes=b"img",
    )
    decision = planner.decide(handoff)
    assert decision.action == PlannerAction.STOP


def test_planner_decide_replan(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubFeedbackEvaluator(FeedbackDecision.REPLAN, "retry")
    monkeypatch.setattr("planner.LLMFeedbackEvaluator", lambda model: stub)
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner("goal", "catalog", [])
    handoff = FeedbackHandoff(
        description="grasp failed",
        prompt="grasp failed",
        image_bytes=b"img",
    )
    decision = planner.decide(handoff)
    assert decision.action == PlannerAction.NEW_PLAN
    assert decision.plan is not None
    assert decision.plan.goal == "pick the cube"
