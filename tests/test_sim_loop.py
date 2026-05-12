"""Hermetic end-to-end regression test for the arm-first sim loop.

No network / OpenAI calls: LLMAgent.llm.invoke is monkeypatched to return
three deterministic tool-call responses (move_ee → set_gripper → finish_task).

Isolation note: test_bug_fixes.py installs sys.modules stubs for langchain_core.*
at collection time. This file uses only plain Python objects (_FakeTool,
_FakeResponse) so it is immune to those stubs.
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

# Hardcoded 1×1 black pixel JPEG – avoids cv2 dependency in test helpers
# (test_bug_fixes.py stubs cv2 globally at collection time).
_MINIMAL_JPEG = (
	b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
	b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
	b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
	b"\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342"
	b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
	b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
	b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
	b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb&\xa2\x8a(\x03\xff\xd9"
)


# ---------------------------------------------------------------------------
# Stubs – plain Python, no langchain_core required
# ---------------------------------------------------------------------------

class _FakeCamera:
	"""Returns a hardcoded minimal JPEG; no cv2 needed."""

	def capture_image(self, **kwargs) -> bytes:
		return _MINIMAL_JPEG


class _FakeTool:
	"""Plain tool stub with a .name and .invoke() — no langchain_core needed."""

	def __init__(self, name: str, return_value: str = "ok") -> None:
		self.name = name
		self.description = f"Fake tool {name}"
		self._return_value = return_value

	def invoke(self, args, **kw) -> str:
		return self._return_value


class _FakeResponse:
	"""Minimal AI-message stub returned by the mocked llm.invoke.

	Only needs .content and .tool_calls to satisfy main_loop_content().
	"""

	type = "ai"  # makes cut_off_context behave correctly if called

	def __init__(self, tool_calls: list[dict]) -> None:
		self.content = ""
		self.tool_calls = tool_calls
		# LLMAgent.main_loop_content may inspect usage_metadata for logging;
		# provide a stub attribute so this fake matches the real response shape.
		self.usage_metadata = None


def _tc(name: str, args: dict, call_id: str) -> dict:
	"""Build a tool-call dict as expected by LLMAgent.invoke_tool."""
	return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _make_responses():
	"""Three-step deterministic plan: move_ee → set_gripper → finish_task."""
	return iter([
		_FakeResponse([_tc("move_ee", {"dx": 0.1, "dy": 0.0, "dz": 0.0}, "id1")]),
		_FakeResponse([_tc("set_gripper", {"open": False}, "id2")]),
		_FakeResponse([_tc("finish_task", {}, "id3")]),
	])


def _build_agent(monkeypatch):
	"""Construct an LLMAgent backed entirely by stubs (no network, no hardware)."""
	mock_llm = MagicMock()
	mock_bound = MagicMock()
	mock_llm.bind_tools.return_value = mock_bound
	monkeypatch.setattr("core.llm_agent.init_chat_model", lambda m: mock_llm)

	from core.llm_agent import LLMAgent

	tools = [
		_FakeTool("move_ee", "Moved EE."),
		_FakeTool("set_gripper", "Gripper closed."),
		_FakeTool("finish_task", "Task finished"),
	]
	camera = _FakeCamera()
	agent = LLMAgent(
		model="fake-model",
		tools=tools,
		main_camera=camera,
		system_prompt="test",
	)
	agent.task = "pick the cube"
	return agent, camera


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loop_exits_early_on_finish_task(monkeypatch):
	"""Loop must stop as soon as finish_task is called (3rd iteration)."""
	agent, _cam = _build_agent(monkeypatch)
	responses = _make_responses()
	agent.llm.invoke = lambda msgs: next(responses)

	loops = 0
	finished = False
	while agent.task:
		result = agent.main_loop_content()
		loops += 1
		if result == "Task finished, going idle.":
			finished = True
			break

	assert finished, "loop must set finished=True when finish_task is called"
	assert loops == 3, f"expected exactly 3 loops, got {loops}"


def test_tool_calls_executed_and_counted(monkeypatch):
	"""All three planned tools are actually invoked by LLMAgent."""
	agent, _cam = _build_agent(monkeypatch)
	responses = _make_responses()
	agent.llm.invoke = lambda msgs: next(responses)

	tool_call_counts: dict[str, int] = {}
	while agent.task:
		result = agent.main_loop_content()
		# Count tool calls from most recent _FakeResponse in history
		for msg in reversed(agent.message_history):
			if isinstance(msg, _FakeResponse) and msg.tool_calls:
				for tc in msg.tool_calls:
					name = tc["name"] if isinstance(tc, dict) else tc.name
					tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
				break
		if result == "Task finished, going idle.":
			break

	assert tool_call_counts.get("move_ee") == 1, tool_call_counts
	assert tool_call_counts.get("set_gripper") == 1, tool_call_counts
	assert tool_call_counts.get("finish_task") == 1, tool_call_counts


def test_run_summary_populated(monkeypatch, tmp_path):
	"""RunSummary fields match expectations after a full 3-step run.

	The run_sim_loop logic is inlined here rather than imported from the
	example module (which has hardware imports in its else-branch).
	"""
	agent, camera = _build_agent(monkeypatch)
	responses = _make_responses()
	agent.llm.invoke = lambda msgs: next(responses)

	@dataclass
	class RunSummary:
		loops_executed: int = 0
		tool_call_counts: dict = dc_field(default_factory=dict)
		finish_task_called: bool = False
		latest_frame_path: Optional[str] = None

	summary = RunSummary()

	while agent.task:
		result = agent.main_loop_content()
		summary.loops_executed += 1
		for msg in reversed(agent.message_history):
			if isinstance(msg, _FakeResponse) and msg.tool_calls:
				for tc in msg.tool_calls:
					name = tc["name"] if isinstance(tc, dict) else tc.name
					summary.tool_call_counts[name] = summary.tool_call_counts.get(name, 0) + 1
				break
		if result == "Task finished, going idle.":
			summary.finish_task_called = True
			break

	assert summary.finish_task_called is True
	assert summary.loops_executed == 3
	assert summary.tool_call_counts.get("move_ee") == 1
	assert summary.tool_call_counts.get("set_gripper") == 1
	assert summary.tool_call_counts.get("finish_task") == 1
	assert summary.latest_frame_path is None  # not set – mirrors save_frames=False
