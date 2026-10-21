"""Hermetic tests for validation/probe_pair_restore.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROBE_PATH = _REPO_ROOT / "validation" / "probe_pair_restore.py"


def _load_probe_module():
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location("probe_pair_restore", _PROBE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DeterministicFake:
    """State depends only on step count since last restore."""

    def __init__(self) -> None:
        self._step = 0
        self._ckpt_step = 0

    def capture_episode_checkpoint(self):
        return {"step": self._step}

    def restore_episode_checkpoint(self, ckpt):
        self._step = int(ckpt["step"])

    def step(self, action):
        self._step += 1
        return False

    def observe(self):
        val = float(self._step)
        blob = f"s{self._step}".encode()
        return ([blob, blob, blob], [val, val])

    def drawer_joint_state(self):
        return {"door": float(self._step) * 0.01}

    def close(self):
        return None


class _LeakyFake(_DeterministicFake):
    """Ghost state accumulates after first restore; affects arm-2-after-dirty."""

    def __init__(self) -> None:
        super().__init__()
        self._restores = 0
        self._ghost = 0.0

    def restore_episode_checkpoint(self, ckpt):
        super().restore_episode_checkpoint(ckpt)
        self._restores += 1

    def step(self, action):
        if self._restores >= 1:
            self._ghost += 0.01
        return super().step(action)

    def observe(self):
        images, state = super().observe()
        if self._restores >= 2:
            state = [float(v) + self._ghost for v in state]
        return (images, state)


def test_probe_exact_on_deterministic_restore(monkeypatch, tmp_path) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe, "SimEnv", lambda *_a, **_k: _DeterministicFake())
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe_pair_restore.py", "--artifact-dir", str(tmp_path),
         "--script-steps", "5", "--dirty-steps", "3"],
    )
    assert probe.main() == 0
    payload = json.loads(
        (tmp_path / "pair_restore_probe.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "EXACT"
    assert payload["colab_acceptance"] is True
    assert payload["post_restore_B_vs_initial"]["first_diverging_step"] is None


def test_probe_clean_only_when_hidden_state_leaks(monkeypatch, tmp_path) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe, "SimEnv", lambda *_a, **_k: _LeakyFake())
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe_pair_restore.py", "--artifact-dir", str(tmp_path),
         "--script-steps", "5", "--dirty-steps", "10"],
    )
    assert probe.main() == 0
    payload = json.loads(
        (tmp_path / "pair_restore_probe.json").read_text(encoding="utf-8"))
    assert payload["verdict"] in ("CLEAN_ONLY", "INEXACT")
    assert payload["B_vs_A"]["first_diverging_step"] is None
    assert payload["C_vs_A"]["first_diverging_step"] is not None
