"""LLM-backed planner for snapshot-grounded plan generation."""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

import config
from executor import ExecutionStop, StopReason
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
- DO NOT compare arm.grab/arm.place outputs to literal strings like "success"; those tools return JSON payload strings. Use feedback nodes after actions.
- Keep plans concise (≤20 call nodes); single-plan step budget applies per execution.
- The plan must be valid JSON matching the schema above.

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


def _shape_goal(goal: str, scene_objects: list[Any] | None) -> str:
    if not scene_objects:
        return goal
    names = ", ".join(str(o) for o in scene_objects)
    return (
        goal
        + "\n\nAvailable scene objects (use these exact names in all skill args): "
        + names
    )


def _history_line(stop: ExecutionStop) -> str:
    reason = stop.reason.value
    if stop.reason == StopReason.FEEDBACK_POINT:
        detail = stop.feedback_description or ""
    elif stop.reason == StopReason.PLAN_DONE:
        detail = str(stop.return_value) if stop.return_value is not None else ""
    elif stop.reason == StopReason.EXECUTION_ERROR:
        detail = stop.error or ""
    else:
        detail = ""
    return f"{reason}: {detail} | steps={stop.steps}"


def _build_feedback_human(
    shaped_goal: str,
    stop: ExecutionStop,
    snapshot: bytes,
    progress_history: list[str],
) -> HumanMessage:
    history_block = ""
    if progress_history:
        lines = "\n".join(f"- {line}" for line in progress_history)
        history_block = f"Progress so far:\n{lines}\n\n"
    text = (
        f"Goal: {shaped_goal}\n\n"
        f"{history_block}"
        f"Feedback checkpoint: {stop.feedback_description}\n"
        f"{stop.feedback_prompt}\n\n"
        "Review progress history and the scene snapshot. Do not repeat failed "
        "approaches. If the goal is achieved, irrecoverable, or no progress "
        "after repeated attempts, respond with a sentinel STOP plan instead "
        "of an executable plan:\n"
        '{"goal":"<same goal>","body":{"kind":"return","value":'
        '{"kind":"literal","value":"STOP: <reason>"}}}\n'
        "Otherwise emit a fresh plan for the goal."
    )
    return _human_with_snapshot(text=text, snapshot=snapshot)


def _build_initial_human(shaped_goal: str, snapshot: bytes) -> HumanMessage:
    text = (
        f"Goal: {shaped_goal}\n\n"
        "Use the scene snapshot as the initial world state when choosing "
        "objects, ordering actions, and feedback checkpoints."
    )
    return _human_with_snapshot(text=text, snapshot=snapshot)


def _human_with_snapshot(text: str, snapshot: bytes) -> HumanMessage:
    image_b64 = base64.b64encode(snapshot).decode("utf-8")
    human_content: list[Any] = [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        },
    ]
    return HumanMessage(content=human_content)


class Planner(Protocol):
    """Plan generation interface."""

    def plan(
        self,
        snapshot: bytes,
        stop: ExecutionStop | None = None,
    ) -> Plan:
        ...


class LLMPlanner:
    """LLM-backed planner with run context fixed at construction."""

    def __init__(
        self,
        goal: str,
        catalog: str,
        scene_objects: list[Any],
    ) -> None:
        self._goal = goal
        self._catalog = catalog
        self._scene_objects = scene_objects
        self._model_name = config.LLM_MODEL
        self._max_plan_retries = config.MAX_PLAN_RETRIES
        self._progress_history: list[str] = []

    def plan(
        self,
        snapshot: bytes,
        stop: ExecutionStop | None = None,
    ) -> Plan:
        """Emit an initial or mid-loop plan AST dict."""
        shaped_goal = _shape_goal(self._goal, self._scene_objects)
        system_content = _SYSTEM_TEMPLATE.format(
            catalog=self._catalog, ast_format=_AST_FORMAT
        )
        if stop is None:
            first_human = _build_initial_human(shaped_goal, snapshot)
        else:
            if stop.feedback_description is None or stop.feedback_prompt is None:
                raise ValueError("feedback stop missing description or prompt")
            self._progress_history.append(_history_line(stop))
            first_human = _build_feedback_human(
                shaped_goal, stop, snapshot, self._progress_history
            )
            logger.info(
                "LLMPlanner: mid-loop replan at feedback checkpoint %r",
                stop.feedback_description,
            )

        llm = _init_planning_llm(self._model_name)

        messages: list[Any] = [
            SystemMessage(content=system_content),
            first_human,
        ]

        for attempt in range(self._max_plan_retries):
            logger.debug(
                "LLMPlanner: attempt %d/%d for goal=%r",
                attempt + 1,
                self._max_plan_retries,
                shaped_goal,
            )
            logger.info(
                "→ LLM (%s, attempt %d/%d): %s",
                self._model_name,
                attempt + 1,
                self._max_plan_retries,
                shaped_goal,
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
                validated_plan = validate_plan(data)
                logger.info("LLMPlanner: valid plan on attempt %d", attempt + 1)
                return validated_plan
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


__all__ = [
    "LLMPlanner",
    "Planner",
]
