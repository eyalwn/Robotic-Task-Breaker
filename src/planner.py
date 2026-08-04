"""LLM-backed planner for snapshot-grounded plan generation."""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

import config
from executor import ExecutionStop, StopReason
from plan import Plan, validate_plan

logger = logging.getLogger(__name__)

_AST_SPEC = """\
PLAN AST (JSON):
  {"goal": <str>, "body": <stmt>}

Statements:
  seq:      {"kind":"seq", "steps":[<stmt>,...]}
  call:     {"kind":"call", "skill":"<name>", "args":{...}}
  feedback: {"kind":"feedback", "description":"<str>"}
  return:   {"kind":"return", "value":<expr>?}

Args/values use {"kind":"literal","value":...} or {"kind":"var","name":"..."}.
Optional advanced: if/while are allowed.

Rules:
- Only call skills from the catalog.
- Use exact object names from the goal/scene context.
- Insert feedback after major actions (grab, place).
- Emit valid JSON.

Example:
{"goal":"grab cube","body":{"kind":"seq","steps":[
  {"kind":"call","skill":"arm.grab","args":{"object_name":{"kind":"literal","value":"cube"}}},
  {"kind":"feedback","description":"Is the cube grasped?"}
]}}\
"""

_SYSTEM_TEMPLATE = """\
You are a robot task planner. Given a natural-language goal and a skill catalog,
emit a structured plan as JSON.

SKILL CATALOG:
{catalog}

{ast_spec}

Use the scene snapshot as the initial world state when choosing objects,
ordering actions, and feedback checkpoints.

When replanning from a feedback checkpoint, review progress history and the
scene snapshot. Do not repeat failed approaches. If the goal is achieved,
irrecoverable, or no progress after repeated attempts, respond with a sentinel
STOP plan instead of an executable plan:
{{"goal":"<same goal>","body":{{"kind":"return","value":{{"kind":"literal","value":"STOP: <reason>"}}}}}}
Otherwise emit a fresh plan for the goal.

Respond with ONLY the plan JSON object — no markdown, no code fences, no explanation.\
"""


class LLMPlanner:
    """LLM-backed planner"""

    def __init__(
        self,
        goal: str,
        catalog: str,
        scene_objects: list[Any],
    ) -> None:
        self._goal = _shape_goal(goal, scene_objects)
        self._catalog = catalog
        self._model_name = config.LLM_MODEL
        self._max_plan_retries = config.MAX_PLAN_RETRIES
        self._progress_history: list[str] = []
        self._llm = _init_planning_llm(self._model_name)

    def plan(
        self,
        snapshot: bytes,
        stop: ExecutionStop | None = None,
    ) -> Plan:
        """Emit a plan AST dict."""
        system_prompt = _SYSTEM_TEMPLATE.format(
            catalog=self._catalog, ast_spec=_AST_SPEC
        )
        if stop is None:
            task_text = f"Goal: {self._goal}\n\n"
        else:
            if stop.feedback_description is None or stop.feedback_prompt is None:
                raise ValueError("feedback stop missing description or prompt")
            self._progress_history.append(_history_line(stop))
            if len(self._progress_history) > config.PROGRESS_HISTORY_WINDOW:
                self._progress_history = self._progress_history[
                    -config.PROGRESS_HISTORY_WINDOW:
                ]
            history_block = ""
            if self._progress_history:
                lines = "\n".join(f"- {line}" for line in self._progress_history)
                history_block = f"Progress so far:\n{lines}\n\n"
            task_text = (
                f"Goal: {self._goal}\n\n"
                f"{history_block}"
                f"Feedback checkpoint: {stop.feedback_description}\n"
                f"{stop.feedback_prompt}\n"
            )
            logger.info(
                "LLMPlanner: mid-loop replan at feedback checkpoint %r",
                stop.feedback_description,
            )

        image_b64 = base64.b64encode(snapshot).decode("utf-8")
        task_prompt: list[Any] = [
            {"type": "text", "text": task_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
        messages: list[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt),
        ]

        for attempt in range(self._max_plan_retries):
            logger.info("planning attempt: %d/%d", attempt + 1, self._max_plan_retries)
            response = self._llm.invoke(messages)
            if response.usage_metadata:
                logger.debug("LLMPlanner usage: %s", response.usage_metadata)

            raw = response.content.strip()
            raw = _extract_json_object(raw)
            logger.debug("raw plan_json from LLM: %s", raw)

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
    logger.info("initiating chat with model: %s", model)
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


__all__ = [
    "LLMPlanner",
]
