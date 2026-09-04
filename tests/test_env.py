"""Hermetic tests for SimEnv factory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from envs.sim_env import SimEnv


def test_sim_env_vlabench_default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SIM_ENV", "vlabench")
    mock_vlabench = MagicMock()
    mock_vlabench.from_track_episode_index.return_value = "vlabench-instance"
    with patch("envs.vlabench_env.VLABenchEnv", mock_vlabench):
        result = SimEnv("select_fruit", 0)
    mock_vlabench.from_track_episode_index.assert_called_once_with("select_fruit", 0)
    assert result == "vlabench-instance"

def test_sim_env_unknown_sim_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SIM_ENV", "isaac")
    with pytest.raises(AssertionError, match="not supported for this env"):
        SimEnv("some_task", 0)
