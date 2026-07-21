"""Unit tests for validation/m7_wiring_probe helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.m7_wiring_probe import (
    collect_env_snapshot,
    eef_displacement,
    evaluate_m7_2,
    write_summary,
)


def test_eef_displacement():
    assert eef_displacement([0.0, 0.0, 0.0], [0.003, 0.004, 0.0]) == 0.005
    assert eef_displacement(None, [1.0, 2.0, 3.0]) is None
    assert eef_displacement([0.0, 0.0], [1.0, 1.0, 1.0]) is None


def test_evaluate_m7_2_pass_and_fail():
    status, reason = evaluate_m7_2(
        completed_without_error=True,
        displacement=0.01,
        min_displacement=0.001,
    )
    assert status == "pass"
    assert "displaced" in reason

    status, reason = evaluate_m7_2(
        completed_without_error=True,
        displacement=0.0001,
        min_displacement=0.001,
    )
    assert status == "fail"
    assert "below threshold" in reason

    status, reason = evaluate_m7_2(
        completed_without_error=False,
        displacement=0.01,
        min_displacement=0.001,
    )
    assert status == "fail"
    assert "exception" in reason


def test_write_summary_and_env_snapshot(tmp_path: Path):
    snapshot = collect_env_snapshot(
        {
            "VLA_POLICY": "smolvla",
            "GITHUB_TOKEN": "secret-value",
            "PERCEPTION_BACKEND": "sim_gt",
        }
    )
    assert snapshot["VLA_POLICY"] == "smolvla"
    assert snapshot["PERCEPTION_BACKEND"] == "sim_gt"
    assert "GITHUB_TOKEN" not in snapshot

    summary = {
        "run_id": "test-run",
        "milestone": "m7",
        "timestamp_utc": "2025-01-01T00:00:00Z",
        "git_branch": "main",
        "git_sha": "abc123",
        "entrypoint": "validation/m7_wiring_probe.py",
        "argv": ["validation/m7_wiring_probe.py", "--gate", "m7.2"],
        "env_snapshot": snapshot,
        "status": "pass",
        "reason": "ok",
        "metrics": {"displacement_m": 0.01},
        "artifacts": {"run_log": "artifacts/colab/m7/test-run/run.log"},
    }
    path = write_summary(tmp_path, summary)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "pass"
    assert loaded["env_snapshot"]["VLA_POLICY"] == "smolvla"
