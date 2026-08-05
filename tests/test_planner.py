"""LLM planner tests with mocked chat model (no OpenAI)."""
from __future__ import annotations

import json

import pytest

from executor import ExecutionSum, StopReason
from planner import LLMPlanner, _extract_json_object, _strip_code_fences
from plan import get_stop_reason

VALID_PLAN = {
    "goal": "pick the cube",
    "body": {
        "kind": "call",
        "skill": "arm.grab",
        "args": {"object_name": {"kind": "literal", "value": "cube"}},
    },
}

STOP_PLAN = {
    "goal": "goal",
    "body": {
        "kind": "return",
        "value": {"kind": "literal", "value": "STOP: goal achieved"},
    },
}

SNAPSHOT = b"img"


def _feedback_sum(
    description: str = "check grasp",
    prompt: str | None = None,
    steps: int = 0,
) -> ExecutionSum:
    return ExecutionSum(
        reason=StopReason.FEEDBACK_POINT,
        steps=steps,
        feedback_description=description,
        feedback_prompt=prompt if prompt is not None else description,
    )


def _human_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return str(part["text"])
    raise AssertionError(f"unexpected human content: {content!r}")


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
    result = planner.plan(SNAPSHOT)
    assert result.goal == "pick the cube"


def test_planner_retry_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _Response("{bad json"),
        _Response(json.dumps(VALID_PLAN)),
    ]
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM(responses))
    monkeypatch.setattr("planner.config.MAX_PLAN_RETRIES", 2)
    planner = LLMPlanner("pick cube", "catalog", [])
    result = planner.plan(SNAPSHOT)
    assert result.goal == "pick the cube"


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
    result = planner.plan(SNAPSHOT)
    assert result.goal == "pick the cube"
    assert len(spy.calls) == 2
    retry_msg = spy.calls[1][-1].content
    assert "never a bare string like 'else_branch'" in retry_msg


def test_planner_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [_Response("{bad"), _Response("{bad")]
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _LLM(responses))
    monkeypatch.setattr("planner.config.MAX_PLAN_RETRIES", 1)
    planner = LLMPlanner("pick cube", "catalog", [])
    with pytest.raises(RuntimeError, match="plan validation failed"):
        planner.plan(SNAPSHOT)


def test_planner_strips_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    fenced = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    monkeypatch.setattr(
        "planner.init_chat_model",
        lambda _model: _LLM([_Response(fenced)]),
    )
    planner = LLMPlanner("pick cube", "catalog", [])
    result = planner.plan(SNAPSHOT)
    assert result.goal == "pick the cube"


def test_planner_shape_goal_with_scene_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner(
        "put items in basket",
        "catalog",
        ["alphabet_soup_1", "cream_cheese_1"],
    )
    planner.plan(SNAPSHOT)
    human_text = _human_text(spy.calls[0][-1].content)
    assert "put items in basket" in human_text
    assert "alphabet_soup_1" in human_text
    assert "Available scene objects" in human_text


def test_planner_shape_goal_without_scene_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("pick cube", "catalog", [])
    planner.plan(SNAPSHOT)
    human_text = _human_text(spy.calls[0][-1].content)
    assert human_text == "Goal: pick cube\n\n"


def test_planner_system_prompt_is_ultra_minimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    catalog = "arm.grab(object_name)\narm.place(object_name, location)"
    planner = LLMPlanner("pick cube", catalog, [])
    planner.plan(SNAPSHOT)
    system_content = spy.calls[0][0].content
    assert "You are a robot task planner" in system_content
    assert catalog in system_content
    assert "Respond with ONLY the plan JSON object" in system_content
    for shape in ('"kind":"seq"', '"kind":"call"', '"kind":"feedback"', '"kind":"return"'):
        assert shape in system_content
    assert '"kind":"literal"' in system_content
    assert '"kind":"var"' in system_content
    assert "Optional advanced: if/while are allowed." in system_content
    assert system_content.count("Example:") == 1
    assert "Only call skills from the catalog." in system_content
    assert "Use exact object names" in system_content
    assert "Insert feedback after major actions" in system_content
    assert "Emit valid JSON." in system_content
    assert "Use the scene snapshot as the initial world state" in system_content
    assert "Do not repeat failed approaches" in system_content
    assert '"STOP: <reason>"' in system_content
    assert "Otherwise emit a fresh plan for the goal." in system_content
    assert "then_branch" not in system_content
    assert "max_iterations" not in system_content
    assert '"kind":"binop"' not in system_content
    assert '"kind":"not"' not in system_content
    assert "DO NOT compare arm.grab" not in system_content
    assert "Multi-object goals" not in system_content
    assert "WRONG:" not in system_content
    assert "state.list_objects" not in system_content


def test_planner_feedback_multimodal_message(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM([_Response(json.dumps(VALID_PLAN))])
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("goal", "catalog", [])
    result = planner.plan(
        b"img-bytes",
        _feedback_sum("check grasp", "Is the object grasped?"),
    )
    assert result.goal == "pick the cube"
    human_msg = spy.calls[0][-1]
    assert isinstance(human_msg.content, list)
    text_part = human_msg.content[0]
    assert text_part["type"] == "text"
    assert "check grasp" in text_part["text"]
    assert "Is the object grasped?" in text_part["text"]
    image_part = human_msg.content[1]
    assert image_part["type"] == "image_url"
    assert "base64," in image_part["image_url"]["url"]


def test_planner_init_llm_once(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls = 0

    def _track_init(_model: str) -> _LLM:
        nonlocal init_calls
        init_calls += 1
        return _mock_llm(VALID_PLAN)

    monkeypatch.setattr("planner.init_chat_model", _track_init)
    planner = LLMPlanner("goal", "catalog", [])
    planner.plan(SNAPSHOT)
    planner.plan(SNAPSHOT, _feedback_sum("step 1"))
    planner.plan(SNAPSHOT, _feedback_sum("step 2"))
    assert init_calls == 1


def test_planner_progress_history_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLLM(
        [_Response(json.dumps(VALID_PLAN)) for _ in range(5)]
    )
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("goal", "catalog", [])
    for i in range(4):
        planner.plan(
            SNAPSHOT,
            _feedback_sum(f"checkpoint {i}", steps=i),
        )
    text_part = spy.calls[3][-1].content[0]
    history_text = text_part["text"]
    assert "checkpoint 0" not in history_text
    assert "checkpoint 1" in history_text
    assert "checkpoint 2" in history_text
    assert "checkpoint 3" in history_text


def test_planner_keeps_progress_history_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SpyLLM(
        [
            _Response(json.dumps(VALID_PLAN)),
            _Response(json.dumps(VALID_PLAN)),
        ]
    )
    monkeypatch.setattr("planner.init_chat_model", lambda _model: spy)
    planner = LLMPlanner("goal", "catalog", [])
    planner.plan(b"img", _feedback_sum("prior step", steps=1))
    planner.plan(b"img-bytes", _feedback_sum("check grasp", "Is the object grasped?"))
    text_part = spy.calls[1][-1].content[0]
    assert "Progress so far:" in text_part["text"]
    assert "feedback_point: prior step" in text_part["text"]
    assert "feedback_point: check grasp" in text_part["text"]


def test_planner_feedback_missing_description_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner("goal", "catalog", [])
    execution_sum = ExecutionSum(reason=StopReason.FEEDBACK_POINT, steps=0)
    with pytest.raises(ValueError, match="feedback execution summary missing"):
        planner.plan(SNAPSHOT, execution_sum)


def test_planner_stop_sentinel_on_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "planner.init_chat_model",
        lambda _model: _LLM([_Response(json.dumps(STOP_PLAN))]),
    )
    planner = LLMPlanner("goal", "catalog", [])
    result = planner.plan(b"img", _feedback_sum("done?", "done?"))
    assert get_stop_reason(result) == "goal achieved"


def test_planner_stop_sentinel_on_initial_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "planner.init_chat_model",
        lambda _model: _LLM([_Response(json.dumps(STOP_PLAN))]),
    )
    planner = LLMPlanner("goal", "catalog", [])
    result = planner.plan(SNAPSHOT)
    assert get_stop_reason(result) == "goal achieved"


def test_planner_feedback_returns_valid_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("planner.init_chat_model", lambda _model: _mock_llm(VALID_PLAN))
    planner = LLMPlanner("goal", "catalog", [])
    result = planner.plan(b"img", _feedback_sum("grasp failed"))
    assert result.goal == "pick the cube"
