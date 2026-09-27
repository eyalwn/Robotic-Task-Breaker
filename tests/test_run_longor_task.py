"""Hermetic tests for examples/run_longor_task.py Colab entrypoint contract."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "examples" / "run_longor_task.py"
_SCRIPT_ARGV = [str(_SCRIPT), "--task-idx", "3"]


def _run_script(*, success: bool) -> int:
    with patch("runner.Runner") as runner_cls, patch.object(sys, "argv", _SCRIPT_ARGV):
        runner_cls.return_value.run.return_value = success
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(_SCRIPT), run_name="__main__")
    return exc.value.code


def test_run_longor_task_pass_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = _run_script(success=True)
    assert code == 0
    out = capsys.readouterr().out
    assert "PASS task='open_drawer' task_idx=3 success=True" in out


def test_run_longor_task_fail_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    code = _run_script(success=False)
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL task='open_drawer' task_idx=3 success=False" in out


def test_run_longor_task_forwards_task_idx() -> None:
    with patch("runner.Runner") as runner_cls, patch.object(sys, "argv", _SCRIPT_ARGV):
        runner_cls.return_value.run.return_value = True
        with pytest.raises(SystemExit):
            runpy.run_path(str(_SCRIPT), run_name="__main__")
    runner_cls.assert_called_once_with("open_drawer", 3)


def test_run_longor_task_construction_error_prints_fail_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("runner.Runner", side_effect=RuntimeError("bad task_idx")), patch.object(
        sys, "argv", _SCRIPT_ARGV
    ):
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(_SCRIPT), run_name="__main__")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL task='open_drawer' task_idx=3" in out
    assert "bad task_idx" in out
