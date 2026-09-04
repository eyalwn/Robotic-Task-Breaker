"""LLM-backed planner for snapshot-grounded linear plan generation."""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

import config
from plan import parse_planner_payload

logger = logging.getLogger(__name__)

_PLAN_SPEC = """\
PLAN FORMAT (JSON):
  [<step>, ...]   OR   {"stop":"<reason>"}

Each step is a flat object with a required "skill" field plus skill args:
  {"skill":"transfer","object_name":"...","target_name":"..."}

Rules:
- Only call skills from the catalog.
- Use a single transfer step when one transfer completes the goal.
- Name objects from the scene snapshot and natural-language goal, not sim ground-truth IDs.
- Emit valid JSON.

Example:
[
  {"skill":"transfer","object_name":"alphabet soup","target_name":"basket"}
]\
"""

_SYSTEM_TEMPLATE = """\
You are a robot task planner. Given a natural-language goal and a skill catalog,
emit a structured plan as JSON.

SKILL CATALOG:
{catalog}

{plan_spec}

Use the scene snapshot as the initial world state when choosing objects and
ordering actions.

If the goal is achieved, irrecoverable, or cannot be attempted, respond with a
sentinel STOP plan instead of an executable plan:
{{"stop":"<reason>"}}
Otherwise emit a step array that achieves the goal.

Respond with ONLY the plan JSON — no markdown, no code fences, no explanation.\
"""


class LLMPlanner:
    """Generate linear step lists from a goal and RGB snapshot."""

    def __init__(self, goal: str, catalog: str) -> None:
        self._goal = goal
        self._catalog = catalog
        self._model_name = config.LLM_MODEL
        self._max_plan_retries = config.MAX_PLAN_RETRIES
        if self._max_plan_retries < 1:
            raise ValueError(
                f"MAX_PLAN_RETRIES must be >= 1, got {self._max_plan_retries}"
            )
        self._llm = _init_planning_llm(self._model_name)

    def plan(self, snapshot: bytes) -> object:
        """Return executable steps or a STOP sentinel dict."""
        system_prompt = _SYSTEM_TEMPLATE.format(
            catalog=self._catalog, plan_spec=_PLAN_SPEC
        )
        task_text = f"Goal: {self._goal}\n\n"

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
            raw = _extract_json(raw)
            logger.info("raw plan_json from LLM: %s", raw)

            try:
                data = json.loads(raw)
                validated_plan = parse_planner_payload(data)
                logger.info("LLMPlanner: valid plan on attempt %d", attempt + 1)
                return validated_plan
            except (json.JSONDecodeError, ValueError) as exc:
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

def _retry_prompt(raw: str, error_msg: str) -> str:
    """Build a repair prompt after JSON or schema validation failure."""
    return (
        "The plan you emitted is invalid.\n"
        f"Invalid plan JSON:\n{raw}\n\n"
        f"Validation error:\n{error_msg}\n\n"
        "Fix the JSON and respond with ONLY the corrected plan JSON."
    )


def _strip_code_fences(raw: str) -> str:
    """Remove markdown code fences if the model wrapped the JSON."""
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n?", "", raw)
        raw = raw.rstrip("`").strip()
    return raw


def _extract_json(raw: str) -> str:
    """Extract the first balanced JSON value from mixed model output."""
    text = _strip_code_fences(raw.strip())
    if not text:
        return text

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
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
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text
def _init_planning_llm(model: str):
    """Initialize the chat model for planning."""
    logger.info("initiating chat with model: %s", model)
    return init_chat_model(model)


__all__ = [
    "LLMPlanner",
]
