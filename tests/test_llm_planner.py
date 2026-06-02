from __future__ import annotations

import json

import pytest

from core.plan import Plan
from core.plan.llm_planner import LLMPlanner

VALID_PLAN = {
    "goal": "pick the cube",
    "body": {
        "kind": "call",
        "skill": "arm.grab",
        "args": {"object_name": {"kind": "literal", "value": "cube"}},
    },
}


class _Response:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.usage_metadata = None


class _BoundLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    def invoke(self, _messages):
        if self._idx >= len(self._responses):
            return self._responses[-1]
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class _Model:
    def __init__(self, responses):
        self._responses = responses

    def bind_tools(self, _tools, tool_choice=None):
        assert tool_choice == "emit_plan"
        return _BoundLLM(self._responses)


def _mock_llm(plan_dict):
    response = _Response(
        [
            {
                "name": "emit_plan",
                "args": {"plan_json": json.dumps(plan_dict)},
                "id": "tc-1",
            }
        ]
    )
    return _Model([response])


def test_planner_success(monkeypatch) -> None:
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner(model="fake")
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)


def test_planner_retry_on_invalid(monkeypatch) -> None:
    responses = [
        _Response(
            [
                {
                    "name": "emit_plan",
                    "args": {"plan_json": "{bad json"},
                    "id": "tc-1",
                }
            ]
        ),
        _Response(
            [
                {
                    "name": "emit_plan",
                    "args": {"plan_json": json.dumps(VALID_PLAN)},
                    "id": "tc-1",
                }
            ]
        ),
    ]
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: _Model(responses))
    planner = LLMPlanner(model="fake", max_plan_retries=2)
    plan = planner.plan("pick cube", "catalog")
    assert isinstance(plan, Plan)


def test_planner_exhausts_retries(monkeypatch) -> None:
    responses = [
        _Response([{"name": "emit_plan", "args": {"plan_json": "{bad"}, "id": "tc-1"}]),
        _Response([{"name": "emit_plan", "args": {"plan_json": "{bad"}, "id": "tc-1"}]),
    ]
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: _Model(responses))
    planner = LLMPlanner(model="fake", max_plan_retries=1)
    with pytest.raises(RuntimeError, match="plan validation failed"):
        planner.plan("pick cube", "catalog")


def test_planner_no_tool_call(monkeypatch) -> None:
    monkeypatch.setattr("core.plan.llm_planner.init_chat_model", lambda _model: _Model([_Response([])]))
    planner = LLMPlanner(model="fake")
    with pytest.raises(RuntimeError, match="did not call emit_plan"):
        planner.plan("pick cube", "catalog")
