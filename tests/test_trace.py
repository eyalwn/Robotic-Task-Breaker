from __future__ import annotations

import json

from core.plan.trace import TraceWriter, _goal_slug


def test_trace_creates_file(tmp_path) -> None:
    writer = TraceWriter(tmp_path, "pick the cube")
    try:
        assert writer.path.exists()
    finally:
        writer.close()


def test_trace_writes_jsonl(tmp_path) -> None:
    writer = TraceWriter(tmp_path, "pick the cube")
    try:
        writer.write({"event": "plan_start"})
        writer.write({"event": "skill_call"})
        writer.write({"event": "plan_end"})
    finally:
        writer.close()

    lines = writer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    events = [json.loads(line) for line in lines]
    assert [item["event"] for item in events] == ["plan_start", "skill_call", "plan_end"]


def test_goal_slug_special_chars() -> None:
    assert _goal_slug("Pick THE cube!!! now") == "pick_the_cube_now"


def test_trace_context_manager(tmp_path) -> None:
    with TraceWriter(tmp_path, "pick cube") as writer:
        writer.write({"event": "x"})
    assert writer._file.closed
