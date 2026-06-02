from __future__ import annotations

from core.plan.feedback import FeedbackDecision
from core.plan.llm_feedback import LLMFeedbackEvaluator


class _Response:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.usage_metadata = None


class _BoundLLM:
    def __init__(self, response):
        self._response = response

    def invoke(self, _messages):
        return self._response


class _Model:
    def __init__(self, response):
        self._response = response

    def bind_tools(self, _tools, tool_choice=None):
        assert tool_choice == "emit_decision"
        return _BoundLLM(self._response)


def _mock_feedback_llm(decision_str):
    return _Model(
        _Response(
            [
                {
                    "name": "emit_decision",
                    "args": {"decision": decision_str, "reason": "test"},
                    "id": "tc-1",
                }
            ]
        )
    )


def test_feedback_continue(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.plan.llm_feedback.init_chat_model",
        lambda _model: _mock_feedback_llm("continue"),
    )
    evaluator = LLMFeedbackEvaluator(model="fake")
    result = evaluator.evaluate(b"img", "desc", "goal")
    assert result == FeedbackDecision.CONTINUE


def test_feedback_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.plan.llm_feedback.init_chat_model",
        lambda _model: _mock_feedback_llm("stop"),
    )
    evaluator = LLMFeedbackEvaluator(model="fake")
    result = evaluator.evaluate(b"img", "desc", "goal")
    assert result == FeedbackDecision.STOP


def test_feedback_replan(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.plan.llm_feedback.init_chat_model",
        lambda _model: _mock_feedback_llm("replan"),
    )
    evaluator = LLMFeedbackEvaluator(model="fake")
    result = evaluator.evaluate(b"img", "desc", "goal")
    assert result == FeedbackDecision.REPLAN


def test_feedback_no_tool_call_defaults_continue(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.plan.llm_feedback.init_chat_model",
        lambda _model: _Model(_Response([])),
    )
    evaluator = LLMFeedbackEvaluator(model="fake")
    result = evaluator.evaluate(b"img", "desc", "goal")
    assert result == FeedbackDecision.CONTINUE


def test_feedback_unknown_decision_defaults_continue(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.plan.llm_feedback.init_chat_model",
        lambda _model: _mock_feedback_llm("maybe"),
    )
    evaluator = LLMFeedbackEvaluator(model="fake")
    result = evaluator.evaluate(b"img", "desc", "goal")
    assert result == FeedbackDecision.CONTINUE
