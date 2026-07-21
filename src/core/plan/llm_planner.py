"""LLM-backed planner: emits plans via direct JSON response with retry on ValidationError."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .schema import Plan, validate_plan

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
  {"kind":"call","skill":"arm.grab","args":{"object_name":{"kind":"literal","value":"apple_1_main"}}},
  {"kind":"feedback","description":"Is the apple grasped?"},
  {"kind":"call","skill":"arm.place","args":{"object_name":{"kind":"literal","value":"apple_1_main"},"location":{"kind":"literal","value":"basket"}}},
  {"kind":"feedback","description":"Is the apple in the basket?"}
]}}

Multi-object goals: cover every object before any terminal feedback/return.
Example (soup then cream cheese — grab→place→feedback per object, then return):
{"goal":"put the alphabet soup can and the cream cheese box in the basket","body":{"kind":"seq","steps":[
  {"kind":"call","skill":"arm.grab","args":{"object_name":{"kind":"literal","value":"alphabet_soup_1_main"}}},
  {"kind":"feedback","description":"Is the alphabet soup can grasped?"},
  {"kind":"call","skill":"arm.place","args":{"object_name":{"kind":"literal","value":"alphabet_soup_1_main"},"location":{"kind":"literal","value":"basket"}}},
  {"kind":"feedback","description":"Is the alphabet soup can in the basket?"},
  {"kind":"call","skill":"arm.grab","args":{"object_name":{"kind":"literal","value":"cream_cheese_1_main"}}},
  {"kind":"feedback","description":"Is the cream cheese box grasped?"},
  {"kind":"call","skill":"arm.place","args":{"object_name":{"kind":"literal","value":"cream_cheese_1_main"},"location":{"kind":"literal","value":"basket"}}},
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
    """Return the best-effort JSON object substring from LLM text."""
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


class LLMPlanner:
    def __init__(self, model: str, max_plan_retries: int = 3) -> None:
        self._model_name = model
        self._max_plan_retries = max_plan_retries

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        """Call the LLM to emit a plan for *goal* given the *catalog*.

        Retries up to *max_plan_retries* times if the plan fails validation.
        """
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
