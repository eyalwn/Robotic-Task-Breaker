"""LLM planner tests with mocked chat model (no OpenAI)."""
from __future__ import annotations

import json

import pytest

from planner import (
    LLMPlanner,
    PlanningHandoff,
    PlanningResult,
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


def _mock_llm(plan_dict: dict) -> _LLM:
    return _LLM([_Response(json.dumps(plan_dict))])


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
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner("pick cube", "catalog", [])
    result = planner.plan()
    assert isinstance(result, PlanningResult)
    assert result.plan is not None
    assert result.plan.goal == "pick the cube"


def test_planner_retry_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _Response("{bad json"),
        _Response(json.dumps(VALID_PLAN)),
    ]
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM(responses))
    monkeypatch.setattr("planner.config.MAX_PLAN_RETRIES", 2)
    planner = LLMPlanner("pick cube", "catalog", [])
    result = planner.plan()
    assert isinstance(result, PlanningResult)
    assert result.plan is not None


def test_planner_retry_prompt_content(monkeypatch: pytest.MonkeyPatch) -> None:
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
    result = planner.plan()
    assert isinstance(result, PlanningResult)
    assert result.plan is not None
    assert len(spy.calls) == 2
    retry_msg = spy.calls[1][-1].content
    assert "never a bare string like 'else_branch'" in retry_msg


def test_planner_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [_Response("{bad"), _Response("{bad")]
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM(responses))
    monkeypatch.setattr("planner.config.MAX_PLAN_RETRIES", 1)
    planner = LLMPlanner("pick cube", "catalog", [])
    with pytest.raises(RuntimeError, match="plan validation failed"):
        planner.plan()


def test_planner_strips_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    fenced = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    monkeypatch.setattr(
        "planner.init_chat_model",
        lambda _model: _LLM([_Response(fenced)]),
    )
    planner = LLMPlanner("pick cube", "catalog", [])
    result = planner.plan()
    assert isinstance(result, PlanningResult)
    assert result.plan is not None


def test_planner_shape_goal_with_scene_objects(monkeypatch: pytest.MonkeyPatch) -> None:
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
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("pick cube", "catalog", [])
    planner.plan()
    human_msg = spy.calls[0][-1].content
    assert human_msg == "Goal: pick cube"


def test_planner_handoff_multimodal_message(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("goal", "catalog", [])
    handoff = PlanningHandoff(
        description="check grasp",
        prompt="Is the object grasped?",
        image_bytes=b"img-bytes",
    )
    result = planner.plan(handoff)
    assert isinstance(result, PlanningResult)
    assert result.plan is not None
    human_msg = spy.calls[0][-1]
    assert isinstance(human_msg.content, list)
    text_part = human_msg.content[0]
    assert text_part["type"] == "text"
    assert "check grasp" in text_part["text"]
    assert "Is the object grasped?" in text_part["text"]
    image_part = human_msg.content[1]
    assert image_part["type"] == "image_url"
    assert "base64," in image_part["image_url"]["url"]


def test_planner_handoff_includes_progress_history(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("goal", "catalog", [])
    handoff = PlanningHandoff(
        description="check grasp",
        prompt="Is the object grasped?",
        image_bytes=b"img-bytes",
        progress_history=("feedback_point: prior step | steps=1",),
    )
    planner.plan(handoff)
    text_part = spy.calls[0][-1].content[0]
    assert "Progress so far:" in text_part["text"]
    assert "feedback_point: prior step" in text_part["text"]


def test_planner_stop_json_on_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    stop_payload = {"stop": True, "reason": "goal achieved"}
    monkeypatch.setattr(
        "planner.init_chat_model",
        lambda _model: _LLM([_Response(json.dumps(stop_payload))]),
    )
    planner = LLMPlanner("goal", "catalog", [])
    handoff = PlanningHandoff(
        description="done?",
        prompt="done?",
        image_bytes=b"img",
    )
    result = planner.plan(handoff)
    assert result.stop is True
    assert result.reason == "goal achieved"
    assert result.plan is None


def test_planner_stop_on_initial_call_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    stop_payload = {"stop": True, "reason": "no"}
    monkeypatch.setattr(
        "planner.init_chat_model",
        lambda _model: _LLM([_Response(json.dumps(stop_payload))]),
    )
    planner = LLMPlanner("goal", "catalog", [])
    with pytest.raises(RuntimeError, match="initial plan call returned stop"):
        planner.plan()


def test_planner_handoff_returns_valid_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner("goal", "catalog", [])
    handoff = PlanningHandoff(
        description="grasp failed",
        prompt="grasp failed",
        image_bytes=b"img",
    )
    result = planner.plan(handoff)
    assert result.plan is not None
    assert result.plan.goal == "pick the cube"
