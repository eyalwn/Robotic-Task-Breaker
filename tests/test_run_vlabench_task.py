"""Hermetic tests for examples/run_vlabench_task.py Colab entrypoint contract."""
from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "examples" / "run_vlabench_task.py"
_SCRIPT_ARGV = [str(_SCRIPT), "--task-idx", "3"]


def _load_script():
    spec = importlib.util.spec_from_file_location("run_vlabench_task", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_script(*, success: bool) -> int:
    with patch("runner.Runner") as runner_cls, patch.object(sys, "argv", _SCRIPT_ARGV):
        runner_cls.return_value.run.return_value = success
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(_SCRIPT), run_name="__main__")
    return exc.value.code


def test_run_vlabench_task_pass_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = _run_script(success=True)
    assert code == 0
    out = capsys.readouterr().out
    assert "PASS task='select_fruit' task_idx=3 success=True" in out


def test_run_vlabench_task_fail_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    code = _run_script(success=False)
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL task='select_fruit' task_idx=3 success=False" in out


def test_run_vlabench_task_forwards_task_idx() -> None:
    with patch("runner.Runner") as runner_cls, patch.object(sys, "argv", _SCRIPT_ARGV):
        runner_cls.return_value.run.return_value = True
        with pytest.raises(SystemExit):
            runpy.run_path(str(_SCRIPT), run_name="__main__")
    runner_cls.assert_called_once_with("select_fruit", 3)


def test_run_vlabench_task_construction_error_prints_fail_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("runner.Runner", side_effect=RuntimeError("bad task_idx")), patch.object(
        sys, "argv", _SCRIPT_ARGV
    ):
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(_SCRIPT), run_name="__main__")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL task='select_fruit' task_idx=3" in out
    assert "bad task_idx" in out


def test_run_vlabench_task_episodes_campaign_runs_consecutive_indices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    runner_calls: list[tuple[str, int]] = []

    def make_runner(task: str, task_idx: int) -> MagicMock:
        runner_calls.append((task, task_idx))
        runner = MagicMock()
        runner.run.return_value = task_idx % 2 == 0
        return runner

    with patch.object(mod, "Runner", side_effect=make_runner), patch.object(
        mod, "configure_logging"
    ), patch.object(
        sys,
        "argv",
        [str(_SCRIPT), "--task", "cook_dishes", "--task-idx", "0", "--episodes", "10"],
    ):
        assert mod.main() == 0

    assert runner_calls == [("cook_dishes", idx) for idx in range(10)]
    out = capsys.readouterr().out
    assert (
        "PASS task='cook_dishes' task_idx=0 episodes=10 successes=5/10 rate=0.50"
        in out
    )


def test_run_vlabench_task_episodes_invalid_count_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    with patch.object(mod, "configure_logging"), patch.object(
        sys,
        "argv",
        [str(_SCRIPT), "--task", "cook_dishes", "--episodes", "0"],
    ):
        assert mod.main() == 1
    out = capsys.readouterr().out
    assert "episodes must be >= 1" in out


def test_run_vlabench_task_episodes_invalid_range_exits_before_runner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    with patch.object(mod, "Runner") as runner_cls, patch.object(
        mod, "configure_logging"
    ), patch.object(
        sys,
        "argv",
        [str(_SCRIPT), "--task", "cook_dishes", "--task-idx", "-1", "--episodes", "2"],
    ):
        assert mod.main() == 1
    runner_cls.assert_not_called()
    out = capsys.readouterr().out
    assert "must be >= 0" in out


def test_run_vlabench_task_episodes_out_of_range_exits_before_runner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    with patch.object(mod, "Runner") as runner_cls, patch.object(
        mod, "configure_logging"
    ), patch.object(
        sys,
        "argv",
        [str(_SCRIPT), "--task", "cook_dishes", "--task-idx", "8", "--episodes", "3"],
    ):
        assert mod.main() == 1
    runner_cls.assert_not_called()
    out = capsys.readouterr().out
    assert "available_config_count=10" in out
