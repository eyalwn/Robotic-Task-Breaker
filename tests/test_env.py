"""Hermetic tests for the SimEnv facade."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from envs.sim_env import SimEnv


def test_sim_env_robocasa_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SIM_ENV", "robocasa")
    mock_robocasa = MagicMock()
    mock_robocasa.from_task.return_value = "robocasa-instance"
    with patch("envs.robocasa_env.RoboCasaEnv", mock_robocasa):
        result = SimEnv("open_drawer", 0)
    mock_robocasa.from_task.assert_called_once_with("open_drawer", 0)
    assert isinstance(result, SimEnv)
    assert result._env == "robocasa-instance"


def test_sim_env_forwards_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SIM_ENV", "robocasa")
    backend = MagicMock()
    backend.steps = 4
    backend.task_description = "open the drawer"
    backend.observe.return_value = ((b"png",), [1.0])
    backend.step.return_value = False
    backend.success.return_value = True
    backend.check_subtask.return_value = True
    mock_robocasa = MagicMock()
    mock_robocasa.from_task.return_value = backend
    step = {"skill": "open", "target_name": "drawer"}
    with patch("envs.robocasa_env.RoboCasaEnv", mock_robocasa):
        env = SimEnv("open_drawer", 0)
    assert env.steps == 4
    assert env.task_description == "open the drawer"
    assert env.observe() == ((b"png",), [1.0])
    assert env.step([0.0]) is False
    assert env.success() is True
    assert env.check_subtask(step) is True
    env.close()
    backend.observe.assert_called_once_with()
    backend.step.assert_called_once_with([0.0])
    backend.check_subtask.assert_called_once_with(step)
    backend.success.assert_called_once_with()
    backend.close.assert_called_once_with()


def test_sim_env_unknown_sim_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SIM_ENV", "isaac")
    with pytest.raises(AssertionError, match="not supported for this env"):
        SimEnv("some_task", 0)
