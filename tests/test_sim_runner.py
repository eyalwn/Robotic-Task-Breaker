"""Hermetic tests for sim.runner — no ManiSkill, no network."""
from __future__ import annotations

import pytest

from sim.frame_saver import FrameSaver
from sim.runner import run_sim_agent

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


class _FakeCamera:
    def capture_image(self, **kw) -> bytes:
        return _MINIMAL_JPEG


class _FakeAgent:
    def __init__(self) -> None:
        self.task: str | None = "pick the cube"
        self._calls = 0

    def main_loop_content(self) -> None:
        self._calls += 1


class _FinishAfterN(_FakeAgent):
    """Sets task=None after *n* calls to simulate finish_task."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self._n = n

    def main_loop_content(self) -> None:
        super().main_loop_content()
        if self._calls >= self._n:
            self.task = None


class _SuccessAtN:
    """Returns True on the *n*-th call to check()."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._calls = 0

    def check(self) -> bool:
        self._calls += 1
        return self._calls >= self._n


class _NeverSuccess:
    def check(self) -> bool:
        return False


def test_exits_on_success(tmp_path):
    result = run_sim_agent(
        agent=_FakeAgent(),
        camera=_FakeCamera(),
        success_checker=_SuccessAtN(3),
        frame_saver=FrameSaver(str(tmp_path)),
        max_loops=10,
    )
    assert result["exit_reason"] == "success"
    assert result["loops_executed"] == 3
    assert result["last_frame_path"].endswith("pickcube_success.jpg")


def test_exits_on_finish_task(tmp_path):
    result = run_sim_agent(
        agent=_FinishAfterN(2),
        camera=_FakeCamera(),
        success_checker=_NeverSuccess(),
        frame_saver=FrameSaver(str(tmp_path)),
        max_loops=10,
    )
    assert result["exit_reason"] == "finish_task"
    assert result["loops_executed"] == 2


def test_exits_on_max_loops(tmp_path):
    result = run_sim_agent(
        agent=_FakeAgent(),
        camera=_FakeCamera(),
        success_checker=_NeverSuccess(),
        frame_saver=FrameSaver(str(tmp_path)),
        max_loops=5,
    )
    assert result["exit_reason"] == "max_loops"
    assert result["loops_executed"] == 5
