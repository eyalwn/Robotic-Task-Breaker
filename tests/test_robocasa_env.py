"""Hermetic tests for RoboCasa env helpers and RoboCasaEnv behavior."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from envs import robocasa_env
from envs.robocasa_env import (
    OPEN_CLOSE_DRAWER_TASK, OPEN_DRAWER_GYM_ENV_ID, OPEN_DRAWER_TASK, STATE_DIMS,
    STATE_KEYS, RoboCasaEnv, _OPEN_CLOSE_SUFFIX, _resolve_gym_env_id,
    _task_description_from_reset_obs)


def test_resolve_gym_env_id_open_close_drawer() -> None:
    assert _resolve_gym_env_id(OPEN_CLOSE_DRAWER_TASK) == OPEN_DRAWER_GYM_ENV_ID


def test_resolve_gym_env_id_unknown_task() -> None:
    with pytest.raises(RuntimeError, match="supports tasks"):
        _resolve_gym_env_id("unknown_task")


def test_horizon_by_task() -> None:
    env = RoboCasaEnv.__new__(RoboCasaEnv)
    env._task = OPEN_DRAWER_TASK
    assert env.horizon == 720
    env._task = OPEN_CLOSE_DRAWER_TASK
    assert env.horizon == 1000


def test_init_env_open_close_drawer_appends_suffix() -> None:
    mock_gym = MagicMock()
    mock_env = MagicMock()
    mock_env.reset.return_value = ({}, {})
    mock_gym.make.return_value = mock_env
    with patch.object(robocasa_env, "_import_robocasa_gym", return_value=mock_gym):
        with patch.object(robocasa_env, "_task_description_from_reset_obs",
                          return_value="open the left drawer"):
            with patch.object(RoboCasaEnv, "_record"):
                env = RoboCasaEnv.__new__(RoboCasaEnv)
                env._init_env(OPEN_CLOSE_DRAWER_TASK)
    assert env.task_description == (f"open the left drawer{_OPEN_CLOSE_SUFFIX}")
    assert env._drawer_was_opened is False


def test_task_description_from_reset_obs_reads_string() -> None:
    obs = {"annotation.human.action.task_description": "  Open drawer  "}
    assert _task_description_from_reset_obs(obs) == "Open drawer"


def test_task_description_from_reset_obs_reads_one_element_sequence() -> None:
    obs = {"annotation.human.action.task_description": ("Pull the drawer",)}
    assert _task_description_from_reset_obs(obs) == "Pull the drawer"


def test_task_description_from_reset_obs_reads_one_element_ndarray() -> None:
    obs = {"annotation.human.action.task_description": np.array(
        ["Open the top drawer"])}
    assert _task_description_from_reset_obs(obs) == "Open the top drawer"


def test_task_description_from_reset_obs_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        _task_description_from_reset_obs({})


def test_task_description_from_reset_obs_raises_when_blank() -> None:
    with pytest.raises(RuntimeError, match="blank"):
        _task_description_from_reset_obs(
            {"annotation.human.action.task_description": "   "})


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


def _robocasa_with_kitchen(
        task: str, opened: bool, *,
        door_state: dict[str, float] | None = None) -> tuple[
            RoboCasaEnv, _OpenDrawerTask]:
    drawer_task = _OpenDrawerTask(opened)
    kitchen = drawer_task
    if door_state is not None:
        kitchen = SimpleNamespace(_check_success=drawer_task._check_success,
                                  drawer=_FakeDrawer(door_state))
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._task = task
    robocasa._drawer_was_opened = False
    robocasa._env = SimpleNamespace(unwrapped=_GrootUnwrapped(kitchen))
    return robocasa, drawer_task


class _FakeDrawer:
    def __init__(self, door_state: dict[str, float]) -> None:
        self._door_state = door_state

    def get_door_state(self, env=None) -> dict[str, float]:
        del env
        return dict(self._door_state)


def test_drawer_oracle_calls_unwrapped_env_check_success() -> None:
    robocasa, task = _robocasa_with_kitchen(OPEN_DRAWER_TASK, True)
    assert robocasa._drawer_open_oracle() is True
    assert task._check_success_calls == 1


def test_drawer_oracle_raises_when_check_success_missing() -> None:
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._env = SimpleNamespace(unwrapped=_GrootUnwrapped(SimpleNamespace()))
    with pytest.raises(AttributeError):
        robocasa._drawer_open_oracle()


def test_check_subtask_open_uses_drawer_oracle() -> None:
    robocasa, task = _robocasa_with_kitchen(OPEN_DRAWER_TASK, False)
    step = {"skill": "open", "object_name": "right drawer"}
    assert robocasa.check_subtask(step) is False
    task.opened = True
    assert robocasa.check_subtask(step) is True


def test_check_subtask_close_on_open_drawer_raises_key_error() -> None:
    robocasa, _task = _robocasa_with_kitchen(
        OPEN_DRAWER_TASK, False, door_state={"door": 0.0})
    with pytest.raises(KeyError, match="unsupported skill"):
        robocasa.check_subtask({"skill": "close", "object_name": "left drawer"})


def test_drawer_close_oracle_true_when_joints_closed() -> None:
    robocasa, _ = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, False, door_state={"door": 0.0})
    assert robocasa._drawer_close_oracle() is True


def test_drawer_close_oracle_false_when_joint_above_threshold() -> None:
    robocasa, _ = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, False, door_state={"door": 0.06})
    assert robocasa._drawer_close_oracle() is False


def test_drawer_joint_state_returns_float_map() -> None:
    robocasa, _ = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, False, door_state={"door": np.float32(0.03)})
    state = robocasa.drawer_joint_state()
    assert state["door"] == pytest.approx(0.03)
    assert isinstance(state["door"], float)


def test_drawer_close_oracle_raises_when_no_door_state() -> None:
    robocasa, _ = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, False, door_state={})
    with pytest.raises(RuntimeError, match="non-empty door state"):
        robocasa._drawer_close_oracle()


def test_drawer_close_oracle_raises_when_drawer_missing() -> None:
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._task = OPEN_CLOSE_DRAWER_TASK
    robocasa._env = SimpleNamespace(unwrapped=_GrootUnwrapped(SimpleNamespace()))
    with pytest.raises(RuntimeError, match="missing drawer fixture"):
        robocasa._drawer_close_oracle()


def test_check_subtask_close_uses_close_oracle() -> None:
    robocasa, _ = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, True, door_state={"door": 0.0})
    step = {"skill": "close", "object_name": "left drawer"}
    assert robocasa.check_subtask(step) is True


def test_check_subtask_unsupported_skill_raises_key_error() -> None:
    robocasa, _ = _robocasa_with_kitchen(OPEN_DRAWER_TASK, False)
    with pytest.raises(KeyError, match="unsupported skill"):
        robocasa.check_subtask({"skill": "grab", "object_name": "drawer"})


def test_success_open_drawer_uses_open_oracle() -> None:
    robocasa, task = _robocasa_with_kitchen(OPEN_DRAWER_TASK, False)
    assert robocasa.success() is False
    task.opened = True
    assert robocasa.success() is True


def test_success_open_close_closed_never_opened_is_false() -> None:
    robocasa, _ = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, False, door_state={"door": 0.0})
    assert robocasa._drawer_was_opened is False
    assert robocasa.success() is False


def test_success_open_close_opened_then_closed_is_true() -> None:
    robocasa, task = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, False, door_state={"door": 0.0})
    robocasa._drawer_was_opened = True
    task.opened = False
    assert robocasa.success() is True


def test_success_open_close_opened_and_still_open_is_false() -> None:
    robocasa, task = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, True, door_state={"door": 0.5})
    robocasa._drawer_was_opened = True
    assert robocasa.success() is False


def test_step_latches_drawer_was_opened_for_open_close_task() -> None:
    robocasa, task = _robocasa_with_kitchen(
        OPEN_CLOSE_DRAWER_TASK, False, door_state={"door": 1.0})
    robocasa._steps = 0
    robocasa._writer = None
    robocasa._last_obs = {}

    def gym_step(_native):
        return {}, 0.0, False, False, {}

    robocasa._env.step = gym_step
    with patch.object(RoboCasaEnv, "_record"):
        robocasa.step([0.0] * robocasa_env.expected_action_dim())
    assert robocasa._drawer_was_opened is False
    task.opened = True
    with patch.object(RoboCasaEnv, "_record"):
        robocasa.step([0.0] * robocasa_env.expected_action_dim())
    assert robocasa._drawer_was_opened is True


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
        VideoWriter_fourcc=_fourcc, VideoWriter=_video_writer, COLOR_RGB2BGR=4,
        cvtColor=lambda frame, code: frame[:, :, ::-1])
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
        with patch.object(robocasa_env, "_task_description_from_reset_obs",
                          return_value="Open drawer"):
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

    def step(self, native: dict[str, np.ndarray]) -> tuple[
            object, float, bool, bool, dict]:
        return self.step_obs, 0.0, False, False, {}


def test_step_stores_obs_returned_by_gym_step() -> None:
    fake = _FakeGymStepEnv()
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._task = OPEN_DRAWER_TASK
    robocasa._env = fake
    robocasa._last_obs = {}
    robocasa._steps = 0
    robocasa._writer = None
    with patch.object(RoboCasaEnv, "_record"):
        done = robocasa.step([0.0] * robocasa_env.expected_action_dim())
    assert done is False
    assert robocasa._last_obs is fake.step_obs
