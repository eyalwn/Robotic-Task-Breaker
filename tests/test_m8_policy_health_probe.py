"""Unit tests for validation/m8_policy_health_probe helpers."""
from __future__ import annotations

from validation.m8_policy_health_probe import evaluate_m8


def _base_result() -> dict:
    return {
        "completed_without_error": True,
        "error": None,
        "initial_dist": 0.24,
        "best_dist": 0.08,
        "improving_steps": 15,
        "close_steps": 6,
        "max_lift": 0.03,
        "task_success": False,
        "success_step": 40,
        "steps_applied": 40,
    }


def test_evaluate_m8_passes_on_healthy_pick_flow():
    status, reason = evaluate_m8(
        result=_base_result(),
        approach_dist_threshold=0.12,
        min_dist_drop=0.06,
        min_improving_steps=8,
        min_lift_delta=0.02,
        max_success_steps=120,
    )
    assert status == "pass"
    assert "policy health confirmed" in reason


def test_evaluate_m8_fails_when_approach_not_decisive():
    result = _base_result()
    result["best_dist"] = 0.19
    result["improving_steps"] = 2
    status, reason = evaluate_m8(
        result=result,
        approach_dist_threshold=0.12,
        min_dist_drop=0.06,
        min_improving_steps=8,
        min_lift_delta=0.02,
        max_success_steps=120,
    )
    assert status == "fail"
    assert "approach not decisive" in reason


def test_evaluate_m8_fails_without_close_command():
    result = _base_result()
    result["close_steps"] = 0
    status, reason = evaluate_m8(
        result=result,
        approach_dist_threshold=0.12,
        min_dist_drop=0.06,
        min_improving_steps=8,
        min_lift_delta=0.02,
        max_success_steps=120,
    )
    assert status == "fail"
    assert "no gripper-close commands" in reason


def test_evaluate_m8_fails_without_lift_or_task_success():
    result = _base_result()
    result["max_lift"] = 0.005
    result["task_success"] = False
    status, reason = evaluate_m8(
        result=result,
        approach_dist_threshold=0.12,
        min_dist_drop=0.06,
        min_improving_steps=8,
        min_lift_delta=0.02,
        max_success_steps=120,
    )
    assert status == "fail"
    assert "no success/lift confirmation" in reason


def test_evaluate_m8_fails_when_success_too_late():
    result = _base_result()
    result["success_step"] = 121
    status, reason = evaluate_m8(
        result=result,
        approach_dist_threshold=0.12,
        min_dist_drop=0.06,
        min_improving_steps=8,
        min_lift_delta=0.02,
        max_success_steps=120,
    )
    assert status == "fail"
    assert "too late" in reason

