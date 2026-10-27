"""LLM planner tests with mocked chat model (no OpenAI)."""
from __future__ import annotations

import json

import pytest

from planner import LLMPlanner, _extract_json, _strip_code_fences
from plan import get_stop_reason, is_stop_plan, parse_planner_payload

VALID_PLAN = [{"skill": "open", "object_name": "right drawer"}]

STOP_PLAN = {"stop": "goal achieved"}

SNAPSHOT = b"img"


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


def _mock_llm(plan_payload: list | dict) -> _LLM:
    return _LLM([_Response(json.dumps(plan_payload))])


def test_strip_code_fences() -> None:
    raw = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    stripped = _strip_code_fences(raw)
    assert stripped.startswith("[")
    assert stripped.endswith("]")


def test_extract_json_from_fenced_output() -> None:
    raw = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    parsed = json.loads(_extract_json(raw))
    assert parsed[0]["skill"] == "open"


def test_extract_json_with_trailing_prose() -> None:
    raw = f"{json.dumps(VALID_PLAN)}\n\nNote: open drawer."
    parsed = json.loads(_extract_json(raw))
    assert parsed[0]["object_name"] == "right drawer"


def test_extract_json_brace_balanced_with_string_braces() -> None:
    raw = 'prefix {"stop": "has } brace"} suffix'
    parsed = json.loads(_extract_json(raw))
    assert parsed["stop"] == "has } brace"


def test_planner_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner("catalog")
    result = planner.plan("pick cube", SNAPSHOT)
    assert result == parse_planner_payload(VALID_PLAN)


def test_planner_invalid_plan_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "planner.init_chat_model", lambda _model: _LLM([_Response("{bad")]))
    planner = LLMPlanner("catalog")
    with pytest.raises(RuntimeError, match="plan validation failed"):
        planner.plan("pick cube", SNAPSHOT)


def test_planner_strips_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    fenced = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    monkeypatch.setattr(
        "planner.init_chat_model", lambda _model: _LLM([_Response(fenced)]))
    planner = LLMPlanner("catalog")
    result = planner.plan("pick cube", SNAPSHOT)
    assert result == parse_planner_payload(VALID_PLAN)


def test_planner_snapshot_image_url_is_png_base64(
        monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("catalog")
    planner.plan("pick cube", SNAPSHOT)
    human_content = spy.calls[0][-1].content
    assert isinstance(human_content, list)
    image_part = next(part for part in human_content
                      if isinstance(part, dict) and part.get("type") == "image_url")
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_planner_goal_used_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("catalog")
    planner.plan("put items in basket", SNAPSHOT)
    human_text = _human_text(spy.calls[0][-1].content)
    assert human_text == "Goal: put items in basket\n\n"
    assert "Available scene objects" not in human_text


def test_planner_system_prompt_is_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    catalog = "open(object_name: str) -> None"
    planner = LLMPlanner(catalog)
    planner.plan("pick cube", SNAPSHOT)
    system_content = spy.calls[0][0].content
    assert "You are a robot task planner" in system_content
    assert catalog in system_content
    assert "Respond with ONLY the plan JSON" in system_content
    assert '"stop"' in system_content
    assert "Only call skills from the catalog." in system_content
    assert "Name objects from the scene snapshot" in system_content
    assert "Emit valid JSON." in system_content
    assert '{"skill":"close"' in system_content
    assert "left drawer" in system_content
    assert "identical phrase" in system_content
    assert "Use the scene snapshot as the initial world state" in system_content
    assert "Otherwise emit a step array that achieves the goal." in system_content
    assert "transfer" not in system_content
    assert '"kind"' not in system_content
    assert '"goal"' not in system_content


def test_planner_init_llm_once(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls = 0

    def _track_init(_model: str) -> _LLM:
        nonlocal init_calls
        init_calls += 1
        return _mock_llm(VALID_PLAN)

    monkeypatch.setattr("planner.init_chat_model", _track_init)
    planner = LLMPlanner("catalog")
    planner.plan("goal", SNAPSHOT)
    planner.plan("goal", SNAPSHOT)
    assert init_calls == 1


def test_planner_stop_sentinel_on_initial_call(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("planner.init_chat_model",
                        lambda _model: _LLM([_Response(json.dumps(STOP_PLAN))]))
    planner = LLMPlanner("catalog")
    result = planner.plan("goal", SNAPSHOT)
    assert is_stop_plan(result)
    assert get_stop_reason(result) == "goal achieved"


def test_planner_usage_logging_omits_provider_detail_fields(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    response = _Response(json.dumps(VALID_PLAN))
    response.usage_metadata = {
        "input_tokens": 505, "output_tokens": 26, "total_tokens": 531,
        "input_token_details": {"audio": 0, "cache_read": 0},
        "output_token_details": {"audio": 0, "reasoning": 0}}
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM([response]))
    planner = LLMPlanner("catalog")
    with caplog.at_level("DEBUG", logger="planner"):
        planner.plan("pick cube", SNAPSHOT)
    usage_logs = [record.getMessage() for record in caplog.records
                  if "LLMPlanner usage:" in record.getMessage()]
    assert usage_logs
    assert "input_tokens" in usage_logs[0]
    assert "output_tokens" in usage_logs[0]
    assert "total_tokens" in usage_logs[0]
    assert "audio" not in usage_logs[0]


def _human_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return str(part["text"])
    raise AssertionError(f"unexpected human content: {content!r}")
