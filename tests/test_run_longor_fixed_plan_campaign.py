"""Hermetic tests for validation/run_longor_fixed_plan_campaign.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "validation" / "run_longor_fixed_plan_campaign.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_longor_fixed_plan_campaign", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_transfer_object_parses_task_description() -> None:
    mod = _load_script()
    assert mod._extract_transfer_object("Put the pear into the plate_seen") == "pear"


def test_extract_transfer_object_rejects_non_transfer_goal() -> None:
    mod = _load_script()
    with pytest.raises(RuntimeError, match="unsupported task instruction format"):
        mod._extract_transfer_object("pick the pear")


def test_fixed_prompt_client_uses_explicit_prompt() -> None:
    mod = _load_script()
    client = mod._FixedPromptVLAClient("Put the pear into the plate")
    with patch.object(mod.VLAClient, "predict", return_value=[[0.0] * 7]) as predict:
        assert client.predict((b"image",), [0.0], "executor-generated task") == [[0.0] * 7]
    predict.assert_called_once_with((b"image",), [0.0], "Put the pear into the plate")


def test_fixed_transfer_keeps_oracle_target_separate_from_prompt() -> None:
    mod = _load_script()
    env = MagicMock(task_description="Put the pear into the plate_seen")
    env.success.return_value = True
    executor = MagicMock()
    executor.run.return_value = SimpleNamespace(
        reason=mod.StopReason.PLAN_DONE,
        steps=12,
        error="",
    )
    with patch.object(mod, "SimEnv", return_value=env), patch.object(
        mod, "Executor", return_value=executor
    ):
        success, steps, detail = mod._run_fixed_transfer_episode(
            task="select_fruit",
            task_idx=1,
            prompt_target="plate",
            max_replans=0,
        )

    assert (success, steps, detail) == (True, 12, "done")
    executor.run.assert_called_once_with(
        [{"skill": "transfer", "object_name": "pear", "target_name": "plate_seen"}]
    )
    assert executor._vla._task_prompt == "Put the pear into the plate"
    env.close.assert_called_once()


def test_campaign_runs_expected_indices_and_counts_successes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    calls: list[tuple[str, int, str, int]] = []

    def fake_run_episode(*, task: str, task_idx: int, prompt_target: str, max_replans: int):
        calls.append((task, task_idx, prompt_target, max_replans))
        return (task_idx % 2 == 0), 10 + task_idx, "done"

    with patch.object(mod, "_run_fixed_transfer_episode", side_effect=fake_run_episode), patch.object(
        mod, "validate_evaluation_episode_range"
    ), patch.object(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--task",
            "select_fruit",
            "--start-idx",
            "1",
            "--episodes",
            "3",
            "--prompt-target",
            "plate_seen",
            "--max-replans",
            "2",
        ],
    ):
        assert mod.main() == 0

    assert calls == [
        ("select_fruit", 1, "plate_seen", 2),
        ("select_fruit", 2, "plate_seen", 2),
        ("select_fruit", 3, "plate_seen", 2),
    ]
    out = capsys.readouterr().out
    assert "PASS task='select_fruit' start_idx=1 episodes=3 prompt_target='plate_seen' successes=1/3 rate=0.33" in out


def test_campaign_requires_target_name() -> None:
    mod = _load_script()
    with patch.object(sys, "argv", [str(_SCRIPT), "--episodes", "2"]):
        with pytest.raises(SystemExit, match="2"):
            mod.main()


def test_campaign_invalid_replans_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script()
    with patch.object(
        sys,
        "argv",
        [str(_SCRIPT), "--prompt-target", "plate", "--max-replans", "-1"],
    ):
        assert mod.main() == 1
    out = capsys.readouterr().out
    assert "max_replans must be >= 0" in out
