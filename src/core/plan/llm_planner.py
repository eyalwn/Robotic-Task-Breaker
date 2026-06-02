"""LLM-backed planner: emits plans via tool-calling with retry on ValidationError."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
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
  feedback: {"kind":"feedback", "description":"<what to evaluate>"}
  return:   {"kind":"return", "value":<expr>?}

Expressions:
  {"kind":"literal", "value":<bool|int|float|str|null>}
  {"kind":"var",     "name":"<var_name>"}
  {"kind":"binop",   "op":"=="|"!="|"<"|"<="|">"|">="|"and"|"or", "left":<expr>, "right":<expr>}
  {"kind":"not",     "operand":<expr>}

Rules:
- Only call skills listed in the catalog.
- Insert "feedback" nodes after major actions (grabbing, placing) to evaluate progress.
- Bind return values with "assign_to" to use in later conditions or the final return.
- Keep plans concise (≤20 call nodes). The executor enforces a 30-step hard budget.
- The plan must be valid JSON matching the schema above.\
"""

_SYSTEM_TEMPLATE = """\
You are a robot task planner. Given a natural-language goal and a skill catalog,
emit a structured plan by calling the emit_plan tool with the plan JSON.

SKILL CATALOG:
{catalog}

{ast_format}
"""


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
            first_human += f"\n\nReplan context: {context}"

        @lc_tool
        def emit_plan(plan_json: str) -> str:
            """Emit the robot task plan as a JSON string matching the Plan AST schema."""
            return plan_json

        llm = init_chat_model(self._model_name)
        bound = llm.bind_tools([emit_plan], tool_choice="emit_plan")

        messages: list[Any] = [
            SystemMessage(content=system_content),
            HumanMessage(content=first_human),
        ]

        for attempt in range(self._max_plan_retries):
            logger.info(
                "LLMPlanner: attempt %d/%d for goal=%r",
                attempt + 1,
                self._max_plan_retries,
                goal,
            )
            response = bound.invoke(messages)
            if response.usage_metadata:
                logger.debug("LLMPlanner usage: %s", response.usage_metadata)

            if not response.tool_calls:
                raise RuntimeError(
                    f"LLM did not call emit_plan (attempt {attempt + 1})"
                )

            plan_json: str = response.tool_calls[0]["args"].get("plan_json", "")
            logger.debug("LLMPlanner raw plan_json: %s", plan_json[:500])

            try:
                data = json.loads(plan_json)
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

                # Feed error back: append the AI response + tool result + correction request.
                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=plan_json,
                        tool_call_id=response.tool_calls[0]["id"],
                    )
                )
                messages.append(
                    HumanMessage(
                        content=(
                            f"The plan you emitted is invalid. Error:\n{error_msg}\n\n"
                            "Please fix the error and call emit_plan again with a corrected plan."
                        )
                    )
                )

        raise RuntimeError("LLMPlanner: unreachable")
