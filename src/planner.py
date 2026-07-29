"""LLM-backed planner and feedback evaluator."""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from pydantic import ValidationError

import config
from executor import (
    AlwaysContinueEvaluator,
    FeedbackDecision,
    FeedbackEvaluator,
    FeedbackOutcome,
)
from plan import Plan, validate_plan

logger = logging.getLogger(__name__)

_AST_FORMAT = """\
PLAN AST (JSON):
  Plan: {"goal": <str>, "body": <stmt>}

Statements — "kind" field is the discriminator:
  seq:      {"kind":"seq", "steps":[<stmt>,...]}
  if:       {"kind":"if", "condition":<expr>, "then_branch":<stmt>, "else_branch":<stmt>?}
  while:    {"kind":"while", "condition":<expr>, "body":<stmt>, "max_iterations":<int>?}
  call:     {"kind":"call", "skill":"<name>", "args":{"<param>":<expr>,...}, "assign_to":"<var>"?}
            WRONG: "args":{"location":"basket","count":2}
            RIGHT: "args":{"location":{"kind":"literal","value":"basket"},"count":{"kind":"literal","value":2}}
  feedback: {"kind":"feedback", "description":"<what to evaluate>"}
  return:   {"kind":"return", "value":<expr>?}

Expressions:
  {"kind":"literal", "value":<bool|int|float|str|null>}
  {"kind":"var",     "name":"<var_name>"}
  {"kind":"binop",   "op":"=="|"!="|"<"|"<="|">"|">="|"and"|"or"|"in", "left":<expr>, "right":<expr>}
  {"kind":"not",     "operand":<expr>}

Rules:
- Use "in" to test list membership: {"kind":"binop","op":"in","left":{"kind":"var","name":"obj"},"right":{"kind":"var","name":"objects"}}
- Only call skills listed in the catalog.
- For listing scene objects, use state.list_objects (not list_objects).
- Insert "feedback" nodes after major actions (grabbing, placing) to evaluate progress.
- Bind return values with "assign_to" to use in later conditions or the final return.
- DO NOT compare arm.grab/arm.place outputs to literal strings like "success"; those tools return JSON payload strings. Use feedback nodes to decide success/failure after actions.
- Keep plans concise (≤20 call nodes). The executor enforces a 30-step hard budget.
- The plan must be valid JSON matching the schema above.

Feedback semantics (executor interprets these — not boolean checks):
- CONTINUE: proceed with the current plan. Does NOT mean "the check passed".
- REPLAN: the check or approach failed; executor will request a new plan.
- STOP: goal achieved or irrecoverable failure; halt execution.

After-action plan structure:
- NEVER emit seq: [grab, feedback, return failure-string] without an if-branch.
- After grab + feedback CONTINUE, continue toward place or the next object.
- Return a failure string only in an explicit else branch or when feedback would STOP.

Example (single object — grab → feedback → place → feedback):
{"goal":"put items in basket","body":{"kind":"seq","steps":[
  {"kind":"call","skill":"arm.grab","args":{"object_name":{"kind":"literal","value":"alphabet_soup_1"}}},
  {"kind":"feedback","description":"Is the alphabet soup can grasped?"},
  {"kind":"call","skill":"arm.place","args":{"object_name":{"kind":"literal","value":"alphabet_soup_1"},"location":{"kind":"literal","value":"basket"}}},
  {"kind":"feedback","description":"Is the alphabet soup can in the basket?"}
]}}

Multi-object goals: cover every object before any terminal feedback/return.
Example (soup then cream cheese — grab→place→feedback per object, then return):
{"goal":"put the alphabet soup can and the cream cheese box in the basket","body":{"kind":"seq","steps":[
  {"kind":"call","skill":"arm.grab","args":{"object_name":{"kind":"literal","value":"alphabet_soup_1"}}},
  {"kind":"feedback","description":"Is the alphabet soup can grasped?"},
  {"kind":"call","skill":"arm.place","args":{"object_name":{"kind":"literal","value":"alphabet_soup_1"},"location":{"kind":"literal","value":"basket"}}},
  {"kind":"feedback","description":"Is the alphabet soup can in the basket?"},
  {"kind":"call","skill":"arm.grab","args":{"object_name":{"kind":"literal","value":"cream_cheese_1"}}},
  {"kind":"feedback","description":"Is the cream cheese box grasped?"},
  {"kind":"call","skill":"arm.place","args":{"object_name":{"kind":"literal","value":"cream_cheese_1"},"location":{"kind":"literal","value":"basket"}}},
  {"kind":"feedback","description":"Is the cream cheese box in the basket?"},
  {"kind":"return","value":{"kind":"literal","value":"done"}}
]}}\
"""

_SYSTEM_TEMPLATE = """\
You are a robot task planner. Given a natural-language goal and a skill catalog,
emit a structured plan as JSON.

SKILL CATALOG:
{catalog}

{ast_format}

Respond with ONLY the plan JSON object — no markdown, no code fences, no explanation.\
"""

_REPLAN_CONTEXT_HINT = """\
Replanning notes:
- The previous plan failed a feedback check (REPLAN). Fix the approach; do not repeat grab→feedback→return-failure sequences.
- CONTINUE means "keep executing this plan", not "verification passed".
- After a successful grab, proceed to place remaining objects — only return failure in an explicit else branch.\
"""


def _retry_prompt(raw: str, error_msg: str) -> str:
    return (
        "The plan you emitted is invalid.\n"
        f"Invalid plan JSON:\n{raw}\n\n"
        f"Validation error:\n{error_msg}\n\n"
        "Fix the JSON and respond with ONLY the corrected plan object.\n"
        "Important: every item in any 'steps' array must be a statement object "
        "with a 'kind' field (never a bare string like 'else_branch')."
    )


def _strip_code_fences(raw: str) -> str:
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n?", "", raw)
        raw = raw.rstrip("`").strip()
    return raw


def _extract_json_object(raw: str) -> str:
    text = _strip_code_fences(raw.strip())
    if not text:
        return text

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]


def _model_supports_json_object(model: str) -> bool:
    name = model.split(":", 1)[-1].lower()
    return name.startswith(("gpt-", "chatgpt-", "o1", "o3"))


def _init_planning_llm(model: str):
    llm = init_chat_model(model)
    if not _model_supports_json_object(model):
        return llm
    try:
        return llm.bind(response_format={"type": "json_object"})
    except (TypeError, ValueError, NotImplementedError) as exc:
        logger.debug("LLMPlanner: json_object mode unavailable for %r: %s", model, exc)
        return llm


class Planner(Protocol):
    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        ...


class LLMPlanner:
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        max_plan_retries: int = config.MAX_PLAN_RETRIES,
    ) -> None:
        self._model_name = model
        self._max_plan_retries = max_plan_retries

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        system_content = _SYSTEM_TEMPLATE.format(
            catalog=catalog, ast_format=_AST_FORMAT
        )
        first_human = f"Goal: {goal}"
        if context:
            first_human += f"\n\nReplan context: {context}\n\n{_REPLAN_CONTEXT_HINT}"

        llm = _init_planning_llm(self._model_name)

        messages: list[Any] = [
            SystemMessage(content=system_content),
            HumanMessage(content=first_human),
        ]

        for attempt in range(self._max_plan_retries):
            logger.debug(
                "LLMPlanner: attempt %d/%d for goal=%r",
                attempt + 1,
                self._max_plan_retries,
                goal,
            )
            logger.info(
                "→ LLM (%s, attempt %d/%d): %s",
                self._model_name,
                attempt + 1,
                self._max_plan_retries,
                goal,
            )
            for i, msg in enumerate(messages):
                logger.debug(
                    "LLMPlanner message[%d] (%s): %s",
                    i,
                    type(msg).__name__,
                    msg.content,
                )
            response = llm.invoke(messages)
            if response.usage_metadata:
                logger.debug("LLMPlanner usage: %s", response.usage_metadata)

            raw = response.content.strip()
            raw = _extract_json_object(raw)
            logger.info("LLMPlanner raw plan_json: %s", raw[:500])

            try:
                data = json.loads(raw)
                plan = validate_plan(data)
                logger.info("LLMPlanner: valid plan on attempt %d", attempt + 1)
                return plan
            except (json.JSONDecodeError, ValidationError) as exc:
                error_msg = str(exc)
                logger.warning(
                    "LLMPlanner: plan invalid on attempt %d: %s",
                    attempt + 1,
                    error_msg[:200],
                )
                if attempt >= self._max_plan_retries - 1:
                    raise RuntimeError(
                        f"LLMPlanner: plan validation failed after "
                        f"{self._max_plan_retries} attempts. "
                        f"Last error: {error_msg}"
                    ) from exc

                messages.append(
                    HumanMessage(
                        content=_retry_prompt(raw=raw, error_msg=error_msg)
                    )
                )

        raise RuntimeError("LLMPlanner: unreachable")


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
    "  reason     — brief justification."
)


class LLMFeedbackEvaluator:
    def __init__(self, model: str = config.LLM_MODEL) -> None:
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


__all__ = [
    "AlwaysContinueEvaluator",
    "FeedbackDecision",
    "FeedbackEvaluator",
    "FeedbackOutcome",
    "LLMFeedbackEvaluator",
    "LLMPlanner",
    "Planner",
]
