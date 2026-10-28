"""Hermetic tests for validation/m4_unpaired_stats.py decision logic."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STATS_PATH = _REPO_ROOT / "validation" / "m4_unpaired_stats.py"


def _load_stats():
    spec = importlib.util.spec_from_file_location("m4_unpaired_stats", _STATS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interim_claim_when_p_below_threshold_and_rate_higher() -> None:
    stats = _load_stats()
    result = stats.evaluate(18, 20, 8, 20, "interim")
    assert result["verdict"] == "claim"
    assert result["claim"] is True
    assert result["p_value_one_sided"] <= 0.01


def test_no_claim_when_sample_below_expected_despite_strong_p() -> None:
    stats = _load_stats()
    result = stats.evaluate(14, 15, 3, 15, "interim")
    assert result["p_value_one_sided"] <= 0.01
    assert result["claim"] is False
    assert result["verdict"] == "no_claim"


def test_interim_no_claim_when_rates_equal() -> None:
    stats = _load_stats()
    result = stats.evaluate(10, 20, 10, 20, "interim")
    assert result["verdict"] == "no_claim"
    assert result["claim"] is False


def test_final_uses_higher_p_threshold() -> None:
    stats = _load_stats()
    result = stats.evaluate(28, 50, 18, 50, "final")
    assert result["p_threshold"] == 0.04
    assert result["expected_episodes_per_arm"] == 50


def test_cli_accepts_direct_counts(capsys: pytest.CaptureFixture[str]) -> None:
    stats = _load_stats()
    argv = [
        "m4_unpaired_stats.py", "--mode", "interim",
        "--robotic-task-breaker-successes", "14",
        "--robotic-task-breaker-total", "20", "--bare-successes", "9",
        "--bare-total", "20"]
    sys.argv = argv
    assert stats.main() == 0
    out = capsys.readouterr().out
    assert "M4 interim:" in out
    assert "no_claim" in out or "claim" in out


def test_cli_reads_summary_files(tmp_path: Path) -> None:
    stats = _load_stats()
    arm_summary = tmp_path / "robotic_task_breaker_summary.json"
    bare = tmp_path / "bare_summary.json"
    arm_summary.write_text(
        json.dumps({"successes": 15, "episodes": 20}) + "\n", encoding="utf-8")
    bare.write_text(
        json.dumps({"successes": 8, "episodes": 20}) + "\n", encoding="utf-8")
    out_path = tmp_path / "verdict.json"
    sys.argv = [
        "m4_unpaired_stats.py", "--mode", "interim",
        "--robotic-task-breaker-summary", str(arm_summary),
        "--bare-summary", str(bare), "--output", str(out_path)]
    assert stats.main() == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["robotic_task_breaker"]["successes"] == 15
    assert payload["bare"]["successes"] == 8
