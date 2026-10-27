"""Hermetic tests for validation/run_bare_policy.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BARE_PATH = _REPO_ROOT / "validation" / "run_bare_policy.py"


def _load_bare():
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location("run_bare_policy", _BARE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEnv:
    task_description = "open the right drawer, then close it"

    def __init__(self) -> None:
        self._steps = 0
        self._horizon = 10
        self.closed = False

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def horizon(self) -> int:
        return self._horizon

    def observe(self):
        return ((b"png", b"png", b"png"), [0.0])

    def step(self, _action: list[float]) -> bool:
        self._steps += 1
        return self._steps >= 3

    def check_subtask(self, step: dict[str, str]) -> bool:
        return step["skill"] == "open"

    def success(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def test_run_episode_writes_summary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bare = _load_bare()
    fake_env = _FakeEnv()

    class _FakeSimEnv:
        def __init__(self, _task: str, _idx: int = 0) -> None:
            self._env = fake_env

        @property
        def steps(self) -> int:
            return fake_env.steps

        @property
        def horizon(self) -> int:
            return fake_env.horizon

        @property
        def task_description(self) -> str:
            return fake_env.task_description

        def observe(self):
            return fake_env.observe()

        def step(self, action: list[float]) -> bool:
            return fake_env.step(action)

        def check_subtask(self, step: dict[str, str]) -> bool:
            return fake_env.check_subtask(step)

        def success(self) -> bool:
            return fake_env.success()

        def close(self) -> None:
            fake_env.close()

    class _FakeClient:
        def predict(self, _images, _state, _prompt):
            return [[0.0]]

    monkeypatch.setattr(bare, "SimEnv", _FakeSimEnv)
    monkeypatch.setattr(bare, "VLAClient", lambda: _FakeClient())
    monkeypatch.setattr(bare, "configure_logging", lambda: None)

    sys.argv = [
        "run_bare_policy.py", "--episodes", "2", "--artifact-dir", str(tmp_path)]
    assert bare.main() == 0
    records = [
        json.loads(line) for line in (tmp_path / "bare_episodes.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 2
    assert all(row["arm"] == "bare" for row in records)
    summary = json.loads(
        (tmp_path / "bare_summary.json").read_text(encoding="utf-8"))
    assert summary["episodes"] == 2
    assert summary["successes"] == 2
