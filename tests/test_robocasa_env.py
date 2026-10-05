"""Hermetic tests for RoboCasa env helpers and RoboCasaEnv behavior."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from envs import robocasa_env
from envs.robocasa_env import (
    STATE_DIMS,
    STATE_KEYS,
    OPEN_DRAWER_GYM_ENV_ID,
    RoboCasaEnv,
    _task_description_from_reset_obs,
)


def test_task_description_from_reset_obs_reads_string() -> None:
    obs = {"annotation.human.action.task_description": "  Open drawer  "}
    assert _task_description_from_reset_obs(obs) == "Open drawer"


def test_task_description_from_reset_obs_reads_one_element_sequence() -> None:
    obs = {"annotation.human.action.task_description": ("Pull the drawer",)}
    assert _task_description_from_reset_obs(obs) == "Pull the drawer"


def test_task_description_from_reset_obs_reads_one_element_ndarray() -> None:
    obs = {
        "annotation.human.action.task_description": np.array(["Open the top drawer"])
    }
    assert _task_description_from_reset_obs(obs) == "Open the top drawer"


def test_task_description_from_reset_obs_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        _task_description_from_reset_obs({})


def test_task_description_from_reset_obs_raises_when_blank() -> None:
    with pytest.raises(RuntimeError, match="blank"):
        _task_description_from_reset_obs(
            {"annotation.human.action.task_description": "   "}
        )


class _GrootUnwrapped:
    def __init__(self, kitchen: object) -> None:
        self.env = kitchen


class _OpenDrawerTask:
    def __init__(self, opened: bool) -> None:
        self.opened = opened
        self._check_success_calls = 0

    def _check_success(self) -> bool:
        self._check_success_calls += 1
        return self.opened


def _robocasa_with_kitchen(opened: bool) -> tuple[RoboCasaEnv, _OpenDrawerTask]:
    task = _OpenDrawerTask(opened)
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._env = SimpleNamespace(unwrapped=_GrootUnwrapped(task))
    return robocasa, task


def test_drawer_oracle_calls_unwrapped_env_check_success() -> None:
    robocasa, task = _robocasa_with_kitchen(True)
    assert robocasa._drawer_open_oracle() is True
    assert task._check_success_calls == 1


def test_drawer_oracle_raises_when_check_success_missing() -> None:
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._env = SimpleNamespace(unwrapped=_GrootUnwrapped(SimpleNamespace()))
    with pytest.raises(AttributeError):
        robocasa._drawer_open_oracle()


def test_check_subtask_open_uses_drawer_oracle() -> None:
    robocasa, task = _robocasa_with_kitchen(False)
    step = {"skill": "open", "target_name": "right drawer"}
    assert robocasa.check_subtask(step) is False
    task.opened = True
    assert robocasa.check_subtask(step) is True


def test_record_uses_policy_camera_frame_and_cv2_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    raw_frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    written: list[np.ndarray] = []
    released = {"called": False}
    writer_init: dict[str, object] = {}
    fourcc_args: list[tuple[str, ...]] = []

    class _Writer:
        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            written.append(frame.copy())

        def release(self) -> None:
            released["called"] = True

    writer = _Writer()

    def _video_writer(
        path: str, fourcc: str, fps: int, size: tuple[int, int]) -> _Writer:
        writer_init["path"] = path
        writer_init["fourcc"] = fourcc
        writer_init["fps"] = fps
        writer_init["size"] = size
        return writer

    def _fourcc(*args: str) -> str:
        fourcc_args.append(args)
        return "mp4v"

    cv2 = SimpleNamespace(
        VideoWriter_fourcc=_fourcc,
        VideoWriter=_video_writer,
        COLOR_RGB2BGR=4,
        cvtColor=lambda frame, code: frame[:, :, ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(robocasa_env.config, "SAVE_VIDEO", True)
    monkeypatch.setattr(robocasa_env.config, "VIDEO_PATH", str(tmp_path / "run.mp4"))

    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._last_obs = {"video.res256_image_side_0": raw_frame}
    robocasa._writer = None
    robocasa._steps = 3
    robocasa._env = SimpleNamespace(close=lambda: None)

    robocasa._record()
    robocasa.close()

    assert fourcc_args == [("m", "p", "4", "v")]
    assert len(written) == 1
    assert np.array_equal(written[0], raw_frame[:, :, ::-1])
    assert writer_init["size"] == (raw_frame.shape[1], raw_frame.shape[0])
    assert writer_init["fps"] == robocasa_env.config.VIDEO_FPS
    assert released["called"] is True


def test_init_env_gym_make_uses_enable_render() -> None:
    mock_gym = MagicMock()
    mock_env = MagicMock()
    mock_env.reset.return_value = ({}, {})
    mock_gym.make.return_value = mock_env
    with patch.object(robocasa_env, "_import_robocasa_gym", return_value=mock_gym):
        with patch.object(
            robocasa_env,
            "_task_description_from_reset_obs",
            return_value="Open drawer",
        ):
            with patch.object(RoboCasaEnv, "_record"):
                env = RoboCasaEnv.__new__(RoboCasaEnv)
                env._init_env("open_drawer")
    mock_gym.make.assert_called_once_with(OPEN_DRAWER_GYM_ENV_ID, enable_render=True)


def test_observe_returns_png_bytes() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    obs = {f"video.{key}": frame for key in robocasa_env.CAMERA_KEYS}
    for key, width in zip(STATE_KEYS, STATE_DIMS):
        obs[f"state.{key}"] = np.zeros(width, dtype=np.float32)
    robocasa._last_obs = obs
    images, _state = robocasa.observe()
    for png in images:
        assert png.startswith(b"\x89PNG")


class _FakeGymStepEnv:
    def __init__(self) -> None:
        self.step_obs = {"from_step": True}

    def step(
        self, native: dict[str, np.ndarray]
    ) -> tuple[object, float, bool, bool, dict]:
        return self.step_obs, 0.0, False, False, {}


def test_step_stores_obs_returned_by_gym_step() -> None:
    fake = _FakeGymStepEnv()
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._env = fake
    robocasa._last_obs = {}
    robocasa._steps = 0
    robocasa._writer = None
    with patch.object(RoboCasaEnv, "_record"):
        done = robocasa.step(
            [0.0] * robocasa_env.expected_action_dim())
    assert done is False
    assert robocasa._last_obs is fake.step_obs
