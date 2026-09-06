"""Hermetic tests for examples/run_vlabench_raw_policy.py baseline entrypoint."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "examples" / "run_vlabench_raw_policy.py"
_SCRIPT_ARGV = [str(_SCRIPT)]


def _load_script():
    spec = importlib.util.spec_from_file_location("run_vlabench_raw_policy", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeBaselineEnv:
    """Minimal env stub for raw-policy baseline loop tests."""

    def __init__(
        self,
        *,
        task_description: str = "select the apple",
        success_after_steps: int | None = None,
        steps: int = 0,
    ) -> None:
        self.task_description = task_description
        self._steps = steps
        self._success_after_steps = success_after_steps
        self.step_calls: list[list[float]] = []
        self.closed = False

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self) -> tuple[tuple[bytes, bytes, bytes], list[float]]:
        return (b"front", b"second", b"wrist"), [0.0] * 7

    def step(self, action: list[float]) -> bool:
        self.step_calls.append(list(action))
        self._steps += 1
        return self.success()

    def success(self) -> bool:
        if self._success_after_steps is None:
            return False
        return self._steps >= self._success_after_steps

    def close(self) -> None:
        self.closed = True


class FakeBaselineVLA:
    """Records predict calls and returns a fixed action chunk."""

    def __init__(self, actions=None) -> None:
        self.actions = actions if actions is not None else [[0.1] * 7]
        self.calls: list[tuple[tuple[bytes, ...], list[float], str]] = []

    def predict(
        self,
        images: tuple[bytes, ...],
        state: list[float],
        task: str,
    ) -> list[list[float]]:
        self.calls.append((images, state, task))
        return self.actions


def test_load_evaluation_episode_configs_reads_first_n() -> None:
    mod = _load_script()
    payload = [{"instructions": ["episode-1"]}, {"instructions": ["episode-2"]}]
    with patch.object(mod, "get_evaluation_episode_configs", return_value=payload) as load_cfg:
        configs = mod.load_evaluation_episode_configs("select_fruit", 2)
    load_cfg.assert_called_once_with("select_fruit")
    assert configs == payload


def test_load_evaluation_episode_configs_raises_when_episode_count_exceeds_available() -> None:
    mod = _load_script()
    payload = [{"instructions": ["episode-1"]}]
    with patch.object(mod, "get_evaluation_episode_configs", return_value=payload):
        with pytest.raises(RuntimeError, match="requested 2 episodes, but evaluation has 1"):
            mod.load_evaluation_episode_configs("select_fruit", 2)


def test_load_evaluation_episode_configs_uses_cook_dishes_evaluation() -> None:
    mod = _load_script()
    with patch.object(
        mod,
        "get_evaluation_episode_configs",
        return_value=[{"task": {"components": []}}] * 10,
    ) as load_cfg:
        configs = mod.load_evaluation_episode_configs("cook_dishes", 10)
    load_cfg.assert_called_once_with("cook_dishes")
    assert len(configs) == 10


def test_main_pass_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    with patch.object(mod, "run_baseline", return_value=(3, 10)), patch.object(
        sys, "argv", _SCRIPT_ARGV
    ):
        assert mod.main() == 0
    out = capsys.readouterr().out
    assert "PASS task='select_fruit' baseline successes=3/10 rate=0.30" in out


def test_main_zero_successes_still_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    with patch.object(mod, "run_baseline", return_value=(0, 10)), patch.object(
        sys, "argv", _SCRIPT_ARGV
    ):
        assert mod.main() == 0
    out = capsys.readouterr().out
    assert "successes=0/10 rate=0.00" in out


def test_main_runtime_error_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    with patch.object(
        mod, "run_baseline", side_effect=RuntimeError("sim blew up")
    ), patch.object(sys, "argv", _SCRIPT_ARGV):
        assert mod.main() == 1
    out = capsys.readouterr().out
    assert "FAIL task='select_fruit'" in out
    assert "sim blew up" in out


def test_run_episode_uses_native_task_description() -> None:
    mod = _load_script()
    env = FakeBaselineEnv(task_description="pick the banana")
    vla = FakeBaselineVLA(actions=[[0.0] * 7])
    with patch("config.MAX_EPISODE_STEPS", 5):
        mod.run_episode(env, vla)
    assert vla.calls[0][2] == "pick the banana"


def test_run_episode_executes_full_returned_chunk() -> None:
    mod = _load_script()
    chunk_size = 5
    actions = [[float(i)] * 7 for i in range(chunk_size)]
    env = FakeBaselineEnv()
    vla = FakeBaselineVLA(actions=actions)
    with patch("config.MAX_EPISODE_STEPS", chunk_size):
        assert mod.run_episode(env, vla) is False
    assert len(vla.calls) == 1
    assert len(env.step_calls) == chunk_size
    assert env.step_calls[-1][0] == float(chunk_size - 1)


def test_run_episode_stops_on_success() -> None:
    mod = _load_script()
    env = FakeBaselineEnv(success_after_steps=2)
    vla = FakeBaselineVLA(actions=[[0.0] * 7] * 3)
    with patch("config.MAX_EPISODE_STEPS", 20):
        assert mod.run_episode(env, vla) is True
    assert len(env.step_calls) == 2


def test_run_baseline_resets_env_each_episode() -> None:
    mod = _load_script()
    episode_configs = [{"instructions": ["a"]}, {"instructions": ["b"]}, {"instructions": ["c"]}]
    env_instances: list[FakeBaselineEnv] = []

    def make_env(task: str, config: dict[str, object]) -> FakeBaselineEnv:
        assert task == "select_fruit"
        assert config in episode_configs
        env = FakeBaselineEnv(success_after_steps=1)
        env_instances.append(env)
        return env

    with patch.object(mod, "load_evaluation_episode_configs", return_value=episode_configs), patch.object(
        mod.VLABenchEnv,
        "from_track_episode",
        side_effect=make_env,
    ), patch.object(mod, "VLAClient", return_value=FakeBaselineVLA()):
        successes, total = mod.run_baseline("select_fruit", 3)

    assert successes == 3
    assert total == 3
    assert len(env_instances) == 3
    assert all(env.closed for env in env_instances)


def test_run_baseline_passes_native_instruction_from_each_episode() -> None:
    mod = _load_script()
    episode_configs = [{"instructions": ["episode-a"]}, {"instructions": ["episode-b"]}]
    instructions = ["pick banana", "pick apple"]
    env_instances = [
        FakeBaselineEnv(task_description=instructions[0], success_after_steps=1),
        FakeBaselineEnv(task_description=instructions[1], success_after_steps=1),
    ]

    with patch.object(mod, "load_evaluation_episode_configs", return_value=episode_configs), patch.object(
        mod.VLABenchEnv,
        "from_track_episode",
        side_effect=env_instances,
    ), patch.object(mod, "VLAClient", return_value=FakeBaselineVLA()) as vla_cls:
        mod.run_baseline("select_fruit", 2)

    vla = vla_cls.return_value
    called_tasks = [call[2] for call in vla.calls]
    assert called_tasks == instructions


def test_run_baseline_shares_one_vla_client() -> None:
    mod = _load_script()
    vla = FakeBaselineVLA()
    with patch.object(
        mod, "load_evaluation_episode_configs", return_value=[{"instructions": ["x"]}, {"instructions": ["y"]}]
    ), patch.object(
        mod.VLABenchEnv,
        "from_track_episode",
        side_effect=[FakeBaselineEnv(), FakeBaselineEnv()],
    ), patch.object(mod, "VLAClient", return_value=vla) as vla_cls:
        mod.run_baseline("select_fruit", 2)

    vla_cls.assert_called_once()
