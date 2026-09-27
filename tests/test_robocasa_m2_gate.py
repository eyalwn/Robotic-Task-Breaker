"""Hermetic tests for RoboCasa M2 gate verdict logic and launcher script."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VERDICT = _REPO_ROOT / "validation" / "robocasa_m2_verdict.py"
_GATE = _REPO_ROOT / "validation" / "run_robocasa_m2_gate.py"
_FIXTURE_LOG = _REPO_ROOT / "validation" / "fixtures" / "m1_rollout_sample.log"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_upstream_rollout_log_passes_at_eight_of_ten() -> None:
    mod = _load_module(_VERDICT, "robocasa_m2_verdict")
    text = _FIXTURE_LOG.read_text(encoding="utf-8")
    successes, note, bools = mod.parse_upstream_rollout_log(text)
    assert successes == 8
    assert "10-bool" in note
    assert bools is not None and len(bools) == 10


def test_stage_a_pass_fail_thresholds() -> None:
    mod = _load_module(_VERDICT, "robocasa_m2_verdict_a")
    pass_verdict = mod.stage_a_from_service_successes(7, 10)
    fail_verdict = mod.stage_a_from_service_successes(6, 10)
    assert pass_verdict.status == "PASS"
    assert fail_verdict.status == "FAIL"


def test_combine_overall_blocked_when_stage_missing() -> None:
    mod = _load_module(_VERDICT, "robocasa_m2_verdict_b")
    blocked = mod.stage_a_blocked("linux only")
    passed = mod.stage_a_from_service_successes(8, 10)
    overall, detail = mod.combine_overall(blocked, passed)
    assert overall == "BLOCKED"
    assert "linux only" in detail


def test_parse_service_raw_stdout() -> None:
    mod = _load_module(_VERDICT, "robocasa_m2_verdict_c")
    text = "PASS task='open_drawer' baseline successes=8/10 rate=0.80\n"
    successes, episodes, note = mod.parse_service_raw_stdout(text)
    assert successes == 8
    assert episodes == 10
    assert "baseline" in note


def test_parse_longor_runner_stdout() -> None:
    mod = _load_module(_VERDICT, "robocasa_m2_verdict_d")
    ok, note = mod.parse_longor_runner_stdout(
        "PASS task='open_drawer' task_idx=0 success=True\n"
    )
    assert ok is True
    fail, _ = mod.parse_longor_runner_stdout(
        "FAIL task='open_drawer' task_idx=0 success=False\n"
    )
    assert fail is False


def test_gate_script_writes_blocked_verdict_without_live_stages(
    tmp_path: Path,
) -> None:
    mod = _load_module(_GATE, "run_robocasa_m2_gate")
    argv = [
        str(_GATE),
        "--output-root",
        str(tmp_path),
        "--run-utc",
        "20260101T000000Z",
        "--upstream-rollout-log",
        str(_FIXTURE_LOG),
    ]
    with patch.object(sys, "argv", argv):
        assert mod.main() == 0
    artifact_dir = tmp_path / "20260101T000000Z"
    payload = json.loads((artifact_dir / "verdict.json").read_text(encoding="utf-8"))
    assert payload["overall"] == "BLOCKED"
    assert payload["stage_a_adapter_parity"]["status"] == "BLOCKED"
    assert payload["stage_b_longor_slice"]["status"] == "BLOCKED"
    assert payload["upstream_primitive_reference"]["successes"] == 8
