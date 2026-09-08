"""Hermetic tests for validation/run_m3_longor.py M3 campaign contract."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "validation" / "run_m3_longor.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_m3_longor", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_m3_longor_campaign_runs_consecutive_indices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    subprocess_calls: list[list[str]] = []

    def fake_run(cmd, cwd, check):
        subprocess_calls.append(cmd)
        return MagicMock(returncode=0 if int(cmd[-1]) % 2 == 0 else 1)

    with patch.object(mod.subprocess, "run", side_effect=fake_run), patch.object(
        mod, "validate_evaluation_episode_range"
    ), patch.object(sys, "argv", [str(_SCRIPT), "10"]):
        assert mod.main() == 0

    assert subprocess_calls == [
        [
            sys.executable,
            str(_REPO_ROOT / "examples" / "run_vlabench_task.py"),
            "--task",
            "cook_dishes",
            "--task-idx",
            str(task_idx),
        ]
        for task_idx in range(10)
    ]
    out = capsys.readouterr().out
    assert (
        "PASS task='cook_dishes' task_idx=0 episodes=10 successes=5/10 rate=0.50"
        in out
    )


def test_run_m3_longor_invalid_episode_count_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    with patch.object(sys, "argv", [str(_SCRIPT), "0"]):
        assert mod.main() == 1
    out = capsys.readouterr().out
    assert "episodes must be >= 1" in out


def test_run_m3_longor_invalid_range_exits_before_subprocess(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    with patch.object(mod.subprocess, "run") as run_cmd, patch.object(
        mod, "validate_evaluation_episode_range",
        side_effect=RuntimeError("must be >= 0"),
    ), patch.object(sys, "argv", [str(_SCRIPT), "2"]):
        assert mod.main() == 1
    run_cmd.assert_not_called()
    out = capsys.readouterr().out
    assert "must be >= 0" in out


def test_run_m3_longor_out_of_range_exits_before_subprocess(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    with patch.object(mod.subprocess, "run") as run_cmd, patch.object(
        mod, "validate_evaluation_episode_range",
        side_effect=RuntimeError("available_config_count=10"),
    ), patch.object(sys, "argv", [str(_SCRIPT), "3"]):
        assert mod.main() == 1
    run_cmd.assert_not_called()
    out = capsys.readouterr().out
    assert "available_config_count=10" in out
