"""Unit tests for validation/m7_wiring_probe helpers."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.m7_wiring_probe import (
    SimGtFallbackError,
    eef_displacement,
    evaluate_m7_2,
    evaluate_m7_3,
    main,
    require_real_perception_backend,
    sim_gt_fallback_guard,
    validate_gate,
    verify_real_perception_wiring,
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


def test_evaluate_m7_3_requires_verified_perception():
    status, reason = evaluate_m7_3(
        completed_without_error=True,
        displacement=0.01,
        min_displacement=0.001,
        perception_verified=False,
        perception_reason="sim-GT path attempted",
    )
    assert status == "fail"
    assert "sim-GT path attempted" in reason

    status, reason = evaluate_m7_3(
        completed_without_error=True,
        displacement=0.01,
        min_displacement=0.001,
        perception_verified=True,
    )
    assert status == "pass"
    assert "displaced" in reason


def test_validate_gate_accepts_m7_2_and_m7_3():
    validate_gate("m7.2")
    validate_gate("m7.3")


def test_validate_gate_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported gate 'm7.4'"):
        validate_gate("m7.4")


def test_require_real_perception_backend_enforces_real(monkeypatch):
    monkeypatch.setenv("PERCEPTION_BACKEND", "real")
    assert require_real_perception_backend() == "real"

    monkeypatch.setenv("PERCEPTION_BACKEND", "sim_gt")
    with pytest.raises(ValueError, match="M7.3 requires PERCEPTION_BACKEND=real"):
        require_real_perception_backend()


def test_sim_gt_fallback_guard_blocks_sim_gt_registration(monkeypatch):
    import sim.libero_skills as libero_skills_mod

    called: list[str] = []

    def _real_register(registry, camera):
        called.append("real")

    monkeypatch.setattr(
        "sim.libero_real_perception.register_libero_real_perception",
        _real_register,
    )

    with sim_gt_fallback_guard():
        with pytest.raises(SimGtFallbackError, match="sim-GT perception registration"):
            libero_skills_mod.register_libero_skills(object(), object())

        from sim.libero_perception import register_libero_perception_skills

        register_libero_perception_skills(
            __import__("core.skills", fromlist=["SkillRegistry"]).SkillRegistry(),
            vec_env=types.SimpleNamespace(),
            camera=types.SimpleNamespace(),
            mode="real",
        )

    assert called == ["real"]


def test_verify_real_perception_wiring_passes_with_real_skills(monkeypatch):
    def _fake_real_register(registry, camera):
        for name in (
            "state.list_objects",
            "perception.detect",
            "perception.locate",
        ):
            registry.register(
                types.SimpleNamespace(name=name, perception_source="real")
            )

    monkeypatch.setattr(
        "sim.libero_real_perception.register_libero_real_perception",
        _fake_real_register,
    )

    result = verify_real_perception_wiring(
        vec_env=types.SimpleNamespace(),
        camera=types.SimpleNamespace(),
    )
    assert result["verified"] is True
    assert result["perception_mode"] == "real"
    assert all(source == "real" for source in result["perception_sources"].values())


def test_verify_real_perception_wiring_fails_on_sim_gt_attempt(monkeypatch):
    import sim.libero_skills as libero_skills_mod

    def _bad_register(registry, *, vec_env, camera, mode=None):
        libero_skills_mod.register_libero_skills(registry, vec_env)
        return "sim_gt"

    monkeypatch.setattr(
        "sim.libero_perception.register_libero_perception_skills",
        _bad_register,
    )

    with pytest.raises(SimGtFallbackError, match="sim-GT perception registration"):
        with sim_gt_fallback_guard():
            verify_real_perception_wiring(
                vec_env=types.SimpleNamespace(),
                camera=types.SimpleNamespace(),
            )


def test_main_m7_3_fails_without_real_backend(monkeypatch, capsys):
    monkeypatch.setenv("PERCEPTION_BACKEND", "sim_gt")
    exit_code = main(["--gate", "m7.3"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "PERCEPTION_BACKEND=real" in captured.out
    assert "M7.3 probe FAIL" in captured.out


def test_main_m7_2_rejects_unsupported_gate(capsys):
    exit_code = main(["--gate", "m7.4"])
    assert exit_code == 2
    assert "Unsupported gate" in capsys.readouterr().err


def test_main_does_not_write_colab_m7_artifacts(monkeypatch, tmp_path: Path):
    artifact_root = ROOT / "artifacts" / "colab" / "m7"
    before = (
        {path.relative_to(artifact_root) for path in artifact_root.rglob("*")}
        if artifact_root.exists()
        else set()
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PERCEPTION_BACKEND", "sim_gt")

    exit_code = main(["--gate", "m7.3"])
    assert exit_code == 1
    assert not (tmp_path / "artifacts" / "colab" / "m7").exists()

    after = (
        {path.relative_to(artifact_root) for path in artifact_root.rglob("*")}
        if artifact_root.exists()
        else set()
    )
    assert before == after
