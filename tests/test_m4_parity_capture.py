"""Tests for M4 parity artifact helpers."""
from __future__ import annotations

import json
from pathlib import Path

from m4_parity_capture import apply_stop_condition, build_metadata, write_artifact


def test_build_metadata_success_rate(tmp_path: Path) -> None:
    payload = build_metadata(
        track="lerobot_baseline",
        n_episodes=10,
        successes=7,
        repo_root=tmp_path,
        log_path=tmp_path / "run.log",
    )
    assert payload["success_rate"] == 0.7
    assert payload["gate"] == "m4_parity"


def test_apply_stop_condition_labels() -> None:
    baseline_ok = {"success_rate": 0.6}
    longor_ok = {"success_rate": 1.0}
    longor_fail = {"success_rate": 0.0}
    baseline_fail = {"success_rate": 0.0}

    assert apply_stop_condition(baseline_ok, longor_ok) == "repaired_longor_succeeds"
    assert (
        apply_stop_condition(baseline_ok, longor_fail)
        == "longor_fails_baseline_succeeds"
    )
    assert apply_stop_condition(baseline_fail, longor_fail) == "official_reference_fails"


def test_write_artifact_roundtrip(tmp_path: Path) -> None:
    payload = build_metadata(
        track="longor_subtask",
        n_episodes=1,
        successes=0,
        repo_root=tmp_path,
        log_path=tmp_path / "run.log",
    )
    out = write_artifact(tmp_path / "out.json", payload)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["track"] == "longor_subtask"
