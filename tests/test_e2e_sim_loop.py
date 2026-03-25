"""End-to-end hermetic sim-loop test.

No ManiSkill, no network — LLM is monkeypatched to emit deterministic tool calls.
Loop sequence: move_ee → set_gripper → finish_task.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from conftest import FakeEnv


# ---------------------------------------------------------------------------
# Fake LLM response
# ---------------------------------------------------------------------------


class _FakeLLMResponse:
    def __init__(self, tool_name: str, args: dict | None = None, tc_id: str = "tc-1"):
        self.content = ""
        self.tool_calls = [{"name": tool_name, "args": args or {}, "id": tc_id}]
        # LLMAgent.main_loop_content may inspect usage_metadata for logging;
        # provide a stub attribute so the fake response matches the real shape.
        self.usage_metadata = None


# ---------------------------------------------------------------------------
# Fake success checker (never fires so exit is finish_task)
# ---------------------------------------------------------------------------


class _NeverSuccess:
    def check(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_e2e_sim_loop(tmp_path, monkeypatch):
    # Monkeypatch init_chat_model so LLMAgent.__init__ doesn't hit the network.
    fake_llm_obj = types.SimpleNamespace(
        bind_tools=lambda tools, **kw: types.SimpleNamespace(invoke=None)
    )
    monkeypatch.setattr(
        "core.LLMAgent.init_chat_model",
        lambda *a, **kw: fake_llm_obj,
    )

    # Import after monkeypatching so the patched name is used.
    from core.LLMAgent import LLMAgent
    from core.tools import finish_task
    from sim.arm_tools import create_move_ee, create_set_gripper
    from sim.frame_saver import FrameSaver
    from sim.maniskill_backend import ManiSkillArm, ManiSkillCamera
    from sim.runner import run_sim_agent

    env = FakeEnv()
    camera = ManiSkillCamera(env)
    arm = ManiSkillArm(env)

    tools = [create_move_ee(arm), create_set_gripper(arm), finish_task]
    agent = LLMAgent(model="fake-model", tools=tools, main_camera=camera)
    agent.task = "pick the cube"

    # Deterministic LLM response sequence: move_ee, set_gripper, finish_task.
    responses = [
        _FakeLLMResponse("move_ee", {"dx": 0.0, "dy": 0.0, "dz": 0.1}, "tc-1"),
        _FakeLLMResponse("set_gripper", {"state": "close"}, "tc-2"),
        _FakeLLMResponse("finish_task", {}, "tc-3"),
    ]
    call_iter = iter(responses)
    agent.llm = types.SimpleNamespace(invoke=lambda msgs: next(call_iter))

    summary = run_sim_agent(
        agent,
        camera,
        _NeverSuccess(),
        FrameSaver(str(tmp_path)),
        max_loops=10,
    )

    # --- assertions ---
    assert summary["exit_reason"] == "finish_task"
    assert summary["loops_executed"] == 3
    assert summary["elapsed_seconds"] >= 0
    assert "last_frame_path" in summary

    # A frame file should have been written and its path returned.
    from pathlib import Path

    assert summary["last_frame_path"] is not None
    assert Path(summary["last_frame_path"]).is_file(), summary["last_frame_path"]
