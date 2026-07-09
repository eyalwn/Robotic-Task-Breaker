"""LLM-backed feedback evaluator: sends an RGB snapshot + description to the LLM."""
from __future__ import annotations

import base64
import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool

from .feedback import FeedbackDecision, FeedbackOutcome

logger = logging.getLogger(__name__)

_SYSTEM = (
    "CRITICAL: 'stop' terminates the ENTIRE run immediately — no further skills execute. "
    "Only emit 'stop' when the overall goal is completely done (every object placed/handled) "
    "OR permanently failed. A positive sub-goal check (e.g. one of two objects placed) "
    "means CONTINUE — proceed to the remaining objects.\n\n"
    "You are a robot task progress evaluator. "
    "Given a camera image and a description of what to check, "
    "answer the check and decide whether the task is progressing well.\n\n"
    "Call emit_decision with:\n"
    "  assessment — your direct answer to the check (e.g. 'yes', 'no', or 'unknown').\n"
    "  decision — one of:\n"
    "    'continue' — sub-goal met or progressing; proceed with remaining plan steps.\n"
    "    'stop'     — ENTIRE goal complete or irrecoverably failed; halt ALL execution.\n"
    "    'replan'   — the current approach is not working; a new plan is needed.\n"
    "  reason     — brief justification.\n\n"
    "If the prior skill result is from arm.place with reason=stagnation, but the "
    "image shows the object in or near the basket, prefer CONTINUE (placement adequate) "
    "over STOP or REPLAN.\n\n"
    "If the evaluation context mentions a prior arm.grab with status=stopped and "
    "grasp_confirmed=false, prefer REPLAN (grab failed — do not proceed to place).\n"
    "If the prior skill result mentions proximity_no_lift or grasp_confirmed=false "
    "with close approach, prefer CONTINUE (retry grasp) over REPLAN."
)


class LLMFeedbackEvaluator:
    def __init__(self, model: str) -> None:
        @lc_tool
        def emit_decision(decision: str, assessment: str, reason: str = "") -> str:
            """Emit feedback: assessment answers the check; decision is continue/stop/replan."""
            return decision

        llm = init_chat_model(model)
        self._llm = llm.bind_tools([emit_decision], tool_choice="emit_decision")

    def evaluate(
        self,
        image_bytes: bytes,
        description: str,
        goal: str,
    ) -> FeedbackOutcome:
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
            return FeedbackOutcome(decision=FeedbackDecision.CONTINUE)

        args = response.tool_calls[0]["args"]
        raw: str = args.get("decision", "continue").lower().strip()
        reason = str(args.get("reason", "") or "")
        assessment = str(args.get("assessment", "") or "").strip() or None
        try:
            decision = FeedbackDecision(raw)
        except ValueError:
            logger.warning(
                "LLMFeedbackEvaluator: unknown decision %r — defaulting to CONTINUE",
                raw,
            )
            decision = FeedbackDecision.CONTINUE

        logger.debug("LLMFeedbackEvaluator full prompt: %s", description)
        return FeedbackOutcome(decision=decision, reason=reason, assessment=assessment)
