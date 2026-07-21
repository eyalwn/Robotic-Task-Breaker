from __future__ import annotations

import json

import pytest

from core.plan import Plan
from core.plan.llm_planner import LLMPlanner, _extract_json_object

VALID_PLAN = {
    "goal": "pick the cube",
    "body": {
        "kind": "call",
        "skill": "arm.grab",
        "args": {"object_name": {"kind": "literal", "value": "cube"}},
    },
}


class _Response:
    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = None


class _LLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    def invoke(self, _messages):
        if self._idx >= len(self._responses):
            return self._responses[-1]
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class _SpyLLM(_LLM):
    def __init__(self, responses):
        super().__init__(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return super().invoke(messages)


def _mock_llm(plan_dict):
    return _LLM([_Response(json.dumps(plan_dict))])


def test_planner_success(monkeypatch) -> None:
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner(model="fake")
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)


def test_planner_retry_on_invalid(monkeypatch) -> None:
    responses = [
        _Response("{bad json"),
        _Response(json.dumps(VALID_PLAN)),
    ]
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: _LLM(responses))
    planner = LLMPlanner(model="fake", max_plan_retries=2)
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)


def test_planner_retry_prompt_does_not_echo_invalid_ai_message(monkeypatch) -> None:
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
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner(model="fake", max_plan_retries=2)
    plan = planner.plan("pick cube", "catalog")

    assert isinstance(plan, Plan)
    assert len(spy.calls) == 2
    second_attempt = spy.calls[1]
    assert all(type(message).__name__ != "AIMessage" for message in second_attempt)
    assert "never a bare string like 'else_branch'" in second_attempt[-1].content


def test_planner_exhausts_retries(monkeypatch) -> None:
    responses = [
        _Response("{bad"),
        _Response("{bad"),
    ]
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: _LLM(responses))
    planner = LLMPlanner(model="fake", max_plan_retries=1)
    with pytest.raises(RuntimeError, match="plan validation failed"):
        planner.plan("pick cube", "catalog")


def test_planner_invalid_json_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.plan.llm_planner.init_chat_model",
        lambda _model: _LLM([_Response("")]),
    )
    planner = LLMPlanner(model="fake", max_plan_retries=1)
    with pytest.raises(RuntimeError, match="plan validation failed"):
        planner.plan("pick cube", "catalog")


@pytest.mark.parametrize("lang", ["json", None])
def test_planner_strips_markdown_fences(monkeypatch, lang) -> None:
    opener = f"```{lang}\n" if lang else "```\n"
    fenced = f"{opener}{json.dumps(VALID_PLAN)}\n```"
    monkeypatch.setattr(
        "core.plan.llm_planner.init_chat_model",
        lambda _model: _LLM([_Response(fenced)]),
    )
    planner = LLMPlanner(model="fake")
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)


@pytest.mark.parametrize(
    ("raw", "expected_goal"),
    [
        (f"```json\n{json.dumps(VALID_PLAN)}\n```", "pick the cube"),
        (
            f"Here is the plan:\n{json.dumps(VALID_PLAN)}\nHope that helps!",
            "pick the cube",
        ),
        (
            json.dumps(VALID_PLAN) + "\n\nNote: use feedback after grabs.",
            "pick the cube",
        ),
    ],
)
def test_extract_json_object_handles_fenced_and_trailing_text(raw, expected_goal) -> None:
    parsed = json.loads(_extract_json_object(raw))
    assert parsed["goal"] == expected_goal


def test_extract_json_object_brace_balanced_with_string_braces() -> None:
    raw = (
        'prefix {"goal": "g", "body": {"kind": "literal", "value": "has } brace"}} '
        "suffix"
    )
    parsed = json.loads(_extract_json_object(raw))
    assert parsed["goal"] == "g"
    assert parsed["body"]["value"] == "has } brace"


def test_planner_parses_json_with_trailing_prose(monkeypatch) -> None:
    wrapped = f"{json.dumps(VALID_PLAN)}\n\nLet me know if you need changes."
    monkeypatch.setattr(
        "core.plan.llm_planner.init_chat_model",
        lambda _model: _LLM([_Response(wrapped)]),
    )
    planner = LLMPlanner(model="fake")
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)


def test_init_planning_llm_binds_json_object_for_openai_models(monkeypatch) -> None:
    class _BoundLLM:
        def invoke(self, _messages):
            return _Response(json.dumps(VALID_PLAN))

    class _BaseLLM:
        bound_kwargs = None

        def bind(self, **kwargs):
            self.bound_kwargs = kwargs
            return _BoundLLM()

    base = _BaseLLM()
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: base)
    planner = LLMPlanner(model="gpt-4o-mini")
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)
    assert base.bound_kwargs == {"response_format": {"type": "json_object"}}
