"""Hermetic tests for RoboCasa task-description resolution helpers."""
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
    _call_check_success,
    _resolve_sim,
    _resolve_task_description,
)


class _DummyEnv:
    """Minimal env stub exposing get_wrapper_attr and unwrapped fields."""

    def __init__(self, attrs: dict[str, object] | None = None) -> None:
        self._attrs = attrs or {}
        self.unwrapped = SimpleNamespace()

    def get_wrapper_attr(self, key: str) -> object:
        if key not in self._attrs:
            raise AttributeError(key)
        return self._attrs[key]


def test_resolve_task_description_prefers_reset_info() -> None:
    env = _DummyEnv({"language": "wrapper language"})
    text = _resolve_task_description(
        "open_drawer",
        env,
        {},
        {"language": "info language"},
    )
    assert text == "info language"


def test_resolve_task_description_reads_obs_annotation() -> None:
    env = _DummyEnv()
    text = _resolve_task_description(
        "open_drawer",
        env,
        {"annotation.human.action.task_description": ("Open drawer from obs",)},
        {},
    )
    assert text == "Open drawer from obs"


def test_resolve_task_description_reads_wrapper_attr() -> None:
    env = _DummyEnv({"language": "wrapper language"})
    text = _resolve_task_description("open_drawer", env, {}, {})
    assert text == "wrapper language"


def test_resolve_task_description_open_drawer_fallback() -> None:
    env = _DummyEnv()
    text = _resolve_task_description("open_drawer", env, {}, {})
    assert text == "Open the drawer"


def test_resolve_task_description_raises_for_unknown_task() -> None:
    env = _DummyEnv()
    with pytest.raises(RuntimeError, match="language instruction is missing"):
        _resolve_task_description("not_supported", env, {}, {})


class _GymForwarder:
    """Mimic gymnasium 0.29 ``Wrapper.__getattr__`` (public names only)."""

    def __init__(self, env: object) -> None:
        self.env = env
        self.forwards = 0

    @property
    def unwrapped(self) -> object:
        return self.env

    def __getattr__(self, name: str) -> object:
        self.forwards += 1
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


class _GrootLayer:
    """Gym env that holds the robosuite task on ``.env`` and has no sim."""

    def __init__(self, env: object) -> None:
        self.env = env

    @property
    def unwrapped(self) -> "_GrootLayer":
        return self


class _OpenDrawerTask:
    def __init__(self, opened: bool) -> None:
        self.opened = opened
        self.sim = SimpleNamespace(data=SimpleNamespace(qpos=[0.0]))

    def _check_success(self) -> bool:
        return self.opened


def _wrapped_open_drawer(opened: bool) -> tuple[RoboCasaEnv, _GymForwarder, _OpenDrawerTask]:
    task = _OpenDrawerTask(opened)
    groot = _GrootLayer(task)
    outer = _GymForwarder(groot)
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._env = outer
    return robocasa, outer, task


def test_drawer_oracle_calls_nested_check_success() -> None:
    robocasa, outer, _task = _wrapped_open_drawer(True)
    assert robocasa._drawer_open_oracle() is True
    assert outer.forwards == 0


def test_call_check_success_raises_when_missing() -> None:
    env = SimpleNamespace(env=SimpleNamespace(unwrapped=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="requires _check_success"):
        _call_check_success(env)


def test_resolve_sim_reads_nested_task_not_wrapper_forwarding() -> None:
    _robocasa, outer, task = _wrapped_open_drawer(False)
    assert _resolve_sim(outer) is task.sim
    assert outer.forwards == 0


def test_resolve_sim_raises_without_sim() -> None:
    env = SimpleNamespace(env=SimpleNamespace(unwrapped=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="requires sim access"):
        _resolve_sim(env)


def test_calc_distances_uses_drawer_site_not_planner_name() -> None:
    robocasa, _outer, task = _wrapped_open_drawer(False)
    task.drawer = SimpleNamespace(naming_prefix="cab_")
    seen: dict[str, str] = {}

    def _site_xpos(site: str) -> np.ndarray:
        seen["site"] = site
        return np.array([0.3, 0.4, 0.0])

    task.sim.data.get_site_xpos = _site_xpos
    obs = {}
    for key, width in zip(STATE_KEYS, STATE_DIMS):
        vec = np.zeros(width, dtype=np.float32)
        if key == "end_effector_position_absolute":
            vec[:] = (0.0, 0.0, 0.0)
        obs[f"state.{key}"] = vec
    robocasa._last_obs = obs
    assert robocasa.calc_distances("EEF", "right drawer") == pytest.approx(0.5)
    assert seen["site"] == "cab_ext_p0"


def test_check_subtask_open_uses_drawer_oracle() -> None:
    robocasa, _outer, task = _wrapped_open_drawer(False)
    step = {"skill": "open", "target_name": "right drawer"}
    assert robocasa.check_subtask(step) is False
    task.opened = True
    assert robocasa.check_subtask(step) is True


def test_record_uses_policy_camera_frame_and_cv2_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
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

    def _video_writer(path: str, fourcc: str, fps: int, size: tuple[int, int]) -> _Writer:
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
    robocasa._steps = 0
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
        with patch.object(robocasa_env, "_resolve_task_description", return_value="Open the drawer"):
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


def test_record_skips_non_quarter_step_indices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    raw_frame = np.zeros((4, 6, 3), dtype=np.uint8)
    written: list[np.ndarray] = []

    class _Writer:
        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            written.append(frame.copy())

        def release(self) -> None:
            pass

    cv2 = SimpleNamespace(
        VideoWriter_fourcc=lambda *a: "mp4v",
        VideoWriter=lambda *a, **k: _Writer(),
        COLOR_RGB2BGR=4,
        cvtColor=lambda frame, code: frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(robocasa_env.config, "SAVE_VIDEO", True)
    monkeypatch.setattr(robocasa_env.config, "VIDEO_PATH", str(tmp_path / "run.mp4"))

    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._last_obs = {"video.res256_image_side_0": raw_frame}
    robocasa._writer = None
    robocasa._env = SimpleNamespace(close=lambda: None)

    for step in (1, 2, 3):
        robocasa._steps = step
        robocasa._record()
    assert len(written) == 0

    for step in (0, 4):
        robocasa._steps = step
        robocasa._record()
    assert len(written) == 2


class _FakeRobosuiteEnv:
    def __init__(self) -> None:
        self.get_obs_calls: list[dict[str, bool]] = []

    def _get_observations(self, force_update: bool = False) -> dict[str, object]:
        self.get_obs_calls.append({"force_update": force_update})
        return {"raw": "obs"}


class _FakeGrootInner:
    def __init__(self) -> None:
        self.env = _FakeRobosuiteEnv()
        self.last_groot: dict[str, object] | None = None

    def get_basic_observation(self, raw: object) -> object:
        return {"basic": raw}

    def get_groot_observation(self, basic: object) -> dict[str, object]:
        self.last_groot = {
            "groot": basic,
            "video.res256_image_side_0": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        return self.last_groot


class _FakeGymStepEnv:
    def __init__(self) -> None:
        self.inner = _FakeGrootInner()
        self.unwrapped = self.inner
        self.step_obs = {"stale": True}

    def step(self, native: dict[str, np.ndarray]) -> tuple[object, float, bool, bool, dict]:
        return self.step_obs, 0.0, False, False, {}


def test_step_last_obs_uses_refreshed_cameras_not_step_return() -> None:
    fake = _FakeGymStepEnv()
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._env = fake
    robocasa._last_obs = {}
    robocasa._steps = 0
    robocasa._writer = None
    with patch.object(RoboCasaEnv, "_record"):
        robocasa.step([0.0] * robocasa_env.expected_action_dim())
    assert robocasa._last_obs is fake.inner.last_groot
    assert robocasa._last_obs is not fake.step_obs
    assert fake.inner.env.get_obs_calls == [{"force_update": True}]
