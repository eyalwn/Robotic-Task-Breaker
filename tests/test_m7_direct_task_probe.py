"""Unit tests for validation/m7_direct_task_probe helpers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.m7_direct_task_probe import (
    evaluate_m7_4,
    main,
    require_real_perception_backend,
)


def test_script_and_module_invocation_help():
    for cmd in (
        [sys.executable, str(ROOT / "validation" / "m7_direct_task_probe.py"), "--help"],
        [sys.executable, "-m", "validation.m7_direct_task_probe", "--help"],
    ):
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "M7.4 direct VLA task probe" in result.stdout


def test_ensure_repo_root_on_path_inserts_repo_root(monkeypatch):
    import validation.m7_direct_task_probe as probe

    root = str(ROOT)
    original_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != root]
        monkeypatch.setattr(probe, "__package__", None, raising=False)
        probe._ensure_repo_root_on_path()
        assert root in sys.path
        assert sys.path[0] == root
    finally:
        sys.path[:] = original_path


def test_evaluate_m7_4_passes_on_env_task_success():
    status, reason = evaluate_m7_4(
        payload={
            "reason": "task_success",
            "status": "success",
            "steps_applied": 17,
        },
        completed_without_error=True,
    )
    assert status == "pass"
    assert "env task success" in reason


def test_evaluate_m7_4_passes_on_explicit_task_success_flag():
    status, reason = evaluate_m7_4(
        payload={
            "task_success": True,
            "status": "success",
            "steps_applied": 9,
        },
        completed_without_error=True,
    )
    assert status == "pass"
    assert "explicit task_success" in reason


@pytest.mark.parametrize(
    "payload",
    [
        {"reason": "grasp_lift", "status": "success", "steps_applied": 5},
        {"reason": "proximity_grasp", "status": "success", "steps_applied": 5},
        {"reason": "episode_done", "status": "stopped", "steps_applied": 3},
        {"reason": "task_success", "status": "stopped", "steps_applied": 3},
    ],
)
def test_evaluate_m7_4_rejects_non_env_success(payload):
    status, reason = evaluate_m7_4(
        payload=payload,
        completed_without_error=True,
    )
    assert status == "fail"
    assert reason != ""
    if payload.get("reason") in {"grasp_lift", "proximity_grasp"}:
        assert "rejected" in reason


def test_evaluate_m7_4_fails_on_exception_or_bad_payload():
    status, reason = evaluate_m7_4(payload=None, completed_without_error=True)
    assert status == "fail"
    assert "non-JSON" in reason

    status, reason = evaluate_m7_4(
        payload={"reason": "task_success", "status": "success"},
        completed_without_error=False,
    )
    assert status == "fail"
    assert "exception" in reason


def test_require_real_perception_backend_enforces_real(monkeypatch):
    monkeypatch.setenv("PERCEPTION_BACKEND", "real")
    assert require_real_perception_backend() == "real"

    monkeypatch.setenv("PERCEPTION_BACKEND", "sim_gt")
    with pytest.raises(ValueError, match="M7.4 requires PERCEPTION_BACKEND=real"):
        require_real_perception_backend()


def test_main_passes_on_task_success(monkeypatch, capsys):
    monkeypatch.setenv("PERCEPTION_BACKEND", "real")
    monkeypatch.setattr(
        "validation.m7_direct_task_probe.run_direct_task_probe",
        lambda **_kwargs: {
            "payload": {
                "reason": "task_success",
                "status": "success",
                "steps_applied": 42,
            },
            "completed_without_error": True,
            "task_prompt": "pick up the object",
            "target_body": "alphabet_soup_1_main",
        },
    )

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "M7.4 probe PASS" in captured.out
    assert "reason=task_success" in captured.out
    assert "steps_applied=42" in captured.out


def test_main_fails_without_real_backend(monkeypatch, capsys):
    monkeypatch.setenv("PERCEPTION_BACKEND", "sim_gt")

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "M7.4 probe FAIL" in captured.out
    assert "PERCEPTION_BACKEND=real" in captured.out


def test_direct_task_probe_does_not_import_libero_smoke():
    source = (ROOT / "validation" / "m7_direct_task_probe.py").read_text(
        encoding="utf-8"
    )
    assert "libero_smoke" not in source


def test_main_fails_on_grasp_lift_success(monkeypatch, capsys):
    monkeypatch.setenv("PERCEPTION_BACKEND", "real")
    monkeypatch.setattr(
        "validation.m7_direct_task_probe.run_direct_task_probe",
        lambda **_kwargs: {
            "payload": {
                "reason": "grasp_lift",
                "status": "success",
                "steps_applied": 10,
            },
            "completed_without_error": True,
            "task_prompt": "pick up the object",
            "target_body": "alphabet_soup_1_main",
        },
    )

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "M7.4 probe FAIL" in captured.out
    assert "rejected" in captured.out
