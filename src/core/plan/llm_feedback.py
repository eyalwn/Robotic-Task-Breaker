"""LLM-backed feedback evaluator: sends an RGB snapshot + description to the LLM."""
from __future__ import annotations

import base64
import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool

from .feedback import FeedbackDecision

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a robot task progress evaluator. "
    "Given a camera image and a description of what to check, "
    "decide whether the task is progressing well.\n\n"
    "Call emit_decision with one of:\n"
    "  'continue' — task is progressing; proceed with the current plan.\n"
    "  'stop'     — task is complete or irrecoverably failed; halt.\n"
    "  'replan'   — the current approach is not working; a new plan is needed.\n\n"
    "Also provide a brief reason."
)


class LLMFeedbackEvaluator:
    def __init__(self, model: str) -> None:
        @lc_tool
        def emit_decision(decision: str, reason: str = "") -> str:
            """Emit a feedback decision: 'continue', 'stop', or 'replan'."""
            return decision

        llm = init_chat_model(model)
        self._llm = llm.bind_tools([emit_decision], tool_choice="emit_decision")

    def evaluate(
        self,
        image_bytes: bytes,
        description: str,
        goal: str,
    ) -> FeedbackDecision:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        human_content: list[Any] = [
            {
                "type": "text",
                "text": f"Overall goal: {goal}\n\nEvaluate: {description}",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=human_content),
        ]
        response = self._llm.invoke(messages)
        if response.usage_metadata:
            logger.debug("LLMFeedbackEvaluator usage: %s", response.usage_metadata)

        if not response.tool_calls:
            logger.warning(
                "LLMFeedbackEvaluator: no tool call — defaulting to CONTINUE"
            )
            return FeedbackDecision.CONTINUE

        raw: str = (
            response.tool_calls[0]["args"].get("decision", "continue").lower().strip()
        )
        try:
            decision = FeedbackDecision(raw)
        except ValueError:
            logger.warning(
                "LLMFeedbackEvaluator: unknown decision %r — defaulting to CONTINUE",
                raw,
            )
            decision = FeedbackDecision.CONTINUE

        logger.info(
            "Feedback [%s] → %s (reason: %s)",
            description,
            decision.value,
            response.tool_calls[0]["args"].get("reason", ""),
        )
        return decision
