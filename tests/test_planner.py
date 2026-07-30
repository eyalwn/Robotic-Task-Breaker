"""LLM planner tests with mocked chat model (no OpenAI)."""
from __future__ import annotations

import json

import pytest

from plan import Plan
from planner import LLMPlanner, _extract_json_object, _strip_code_fences

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
    planner = LLMPlanner(model="fake")
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)
    assert plan.goal == "pick the cube"


def test_planner_retry_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _Response("{bad json"),
        _Response(json.dumps(VALID_PLAN)),
    ]
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM(responses))
    planner = LLMPlanner(model="fake", max_plan_retries=2)
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)


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
    planner = LLMPlanner(model="fake", max_plan_retries=2)
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)
    assert len(spy.calls) == 2
    retry_msg = spy.calls[1][-1].content
    assert "never a bare string like 'else_branch'" in retry_msg


def test_planner_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [_Response("{bad"), _Response("{bad")]
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM(responses))
    planner = LLMPlanner(model="fake", max_plan_retries=1)
    with pytest.raises(RuntimeError, match="plan validation failed"):
        planner.plan("pick cube", "catalog")


def test_planner_strips_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    fenced = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    monkeypatch.setattr(
        "planner.init_chat_model",
        lambda _model: _LLM([_Response(fenced)]),
    )
    planner = LLMPlanner(model="fake")
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)
