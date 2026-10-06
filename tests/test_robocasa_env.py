"""Hermetic tests for RoboCasa env helpers and RoboCasaEnv behavior."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from envs import robocasa_env
from envs.robocasa_env import (
    KETTLE_BOILING_GYM_ENV_ID,
    KETTLE_BOILING_TASK,
    OPEN_DRAWER_GYM_ENV_ID,
    OPEN_DRAWER_TASK,
    STATE_DIMS,
    STATE_KEYS,
    RoboCasaEnv,
    _resolve_gym_env_id,
    _task_description_from_reset_obs,
    burner_phrase_to_key,
)


def test_burner_phrase_to_key_parses_stove_phrase() -> None:
    keys = ("front_left", "front_right", "rear_left", "rear_right")
    assert burner_phrase_to_key(
        "rear right burner of the stove", keys) == "rear_right"


def test_burner_phrase_to_key_unknown_lists_valid_keys() -> None:
    keys = ("rear_right", "front_left")
    with pytest.raises(KeyError, match="valid keys: front_left, rear_right"):
        burner_phrase_to_key("front right burner of the stove", keys)


def test_resolve_gym_env_id_kettle_boiling() -> None:
    assert _resolve_gym_env_id(KETTLE_BOILING_TASK) == KETTLE_BOILING_GYM_ENV_ID


def test_resolve_gym_env_id_unknown_task() -> None:
    with pytest.raises(RuntimeError, match="supports tasks"):
        _resolve_gym_env_id("unknown_task")


def test_horizon_by_task() -> None:
    env = RoboCasaEnv.__new__(RoboCasaEnv)
    env._task = OPEN_DRAWER_TASK
    assert env.horizon == 720
    env._task = KETTLE_BOILING_TASK
    assert env.horizon == 1500


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
    robocasa._task = OPEN_DRAWER_TASK
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
    step = {"skill": "open", "object_name": "right drawer"}
    assert robocasa.check_subtask(step) is False
    task.opened = True
    assert robocasa.check_subtask(step) is True


_BURNER_PHRASE = "rear right burner of the stove"
_BURNER_SITES_OK = {"rear_right": {"name": "rear_right_site"}}
_KETTLE_PLACE_STEP = {
    "skill": "place",
    "object_name": "kettle",
    "source_name": "counter",
    "destination_name": _BURNER_PHRASE,
}
_KETTLE_TURN_STEP = {"skill": "turn_on", "target_name": _BURNER_PHRASE}


def _patch_fake_object_utils(
    monkeypatch: pytest.MonkeyPatch, *, contact: bool, gripper_far: bool
) -> None:
    ou = SimpleNamespace(
        check_obj_fixture_contact=lambda _sim, _obj, _fix: contact,
        gripper_obj_far=lambda _sim: gripper_far,
    )
    utils = SimpleNamespace(object_utils=ou)
    monkeypatch.setitem(sys.modules, "robocasa", SimpleNamespace(utils=utils))
    monkeypatch.setitem(sys.modules, "robocasa.utils", utils)
    monkeypatch.setitem(sys.modules, "robocasa.utils.object_utils", ou)


def _kettle_kitchen(
    *,
    burner_sites: dict,
    kettle_xy: tuple[float, float] = (0.0, 0.0),
    site_xy: tuple[float, float] = (0.0, 0.0),
    knobs: dict[str, float] | None = None,
) -> SimpleNamespace:
    if knobs is None:
        knobs = {"rear_right": 1.0}
    body_xpos = np.zeros((1, 3), dtype=np.float64)
    body_xpos[0, 0:2] = kettle_xy
    site_pos = np.zeros(3, dtype=np.float64)
    site_pos[0:2] = site_xy

    sim_data = SimpleNamespace(
        body_xpos=body_xpos,
        get_site_xpos=lambda _name: site_pos,
    )

    return SimpleNamespace(
        stove=SimpleNamespace(
            burner_sites=burner_sites,
            get_knobs_state=lambda env=None: knobs,
        ),
        objects={"obj": SimpleNamespace(name="kettle_obj")},
        obj_body_id={"kettle_obj": 0},
        sim=SimpleNamespace(data=sim_data),
    )


def _kettle_robocasa(kitchen: SimpleNamespace) -> RoboCasaEnv:
    robocasa = RoboCasaEnv.__new__(RoboCasaEnv)
    robocasa._task = KETTLE_BOILING_TASK
    robocasa._env = SimpleNamespace(unwrapped=_GrootUnwrapped(kitchen))
    return robocasa


def test_kettle_place_oracle_true_when_all_conditions_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_object_utils(monkeypatch, contact=True, gripper_far=True)
    rc = _kettle_robocasa(_kettle_kitchen(burner_sites=_BURNER_SITES_OK))
    assert rc._kettle_place_oracle(_KETTLE_PLACE_STEP) is True


def test_kettle_place_oracle_false_without_fixture_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_object_utils(monkeypatch, contact=False, gripper_far=True)
    rc = _kettle_robocasa(_kettle_kitchen(burner_sites=_BURNER_SITES_OK))
    assert rc._kettle_place_oracle(_KETTLE_PLACE_STEP) is False


def test_kettle_place_oracle_false_when_kettle_too_far_from_burner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_object_utils(monkeypatch, contact=True, gripper_far=True)
    kitchen = _kettle_kitchen(
        burner_sites=_BURNER_SITES_OK,
        kettle_xy=(0.0, 0.0),
        site_xy=(1.0, 0.0),
    )
    rc = _kettle_robocasa(kitchen)
    assert rc._kettle_place_oracle(_KETTLE_PLACE_STEP) is False


def test_kettle_place_oracle_false_when_gripper_not_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_object_utils(monkeypatch, contact=True, gripper_far=False)
    rc = _kettle_robocasa(_kettle_kitchen(burner_sites=_BURNER_SITES_OK))
    assert rc._kettle_place_oracle(_KETTLE_PLACE_STEP) is False


def test_kettle_place_oracle_false_when_burner_site_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_object_utils(monkeypatch, contact=True, gripper_far=True)
    kitchen = _kettle_kitchen(burner_sites={"rear_right": None})
    rc = _kettle_robocasa(kitchen)
    assert rc._kettle_place_oracle(_KETTLE_PLACE_STEP) is False


def test_kettle_place_oracle_bad_burner_phrase_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_object_utils(monkeypatch, contact=True, gripper_far=True)
    rc = _kettle_robocasa(_kettle_kitchen(burner_sites=_BURNER_SITES_OK))
    step = {**_KETTLE_PLACE_STEP, "destination_name": "left drawer"}
    with pytest.raises(ValueError, match="burner phrase"):
        rc._kettle_place_oracle(step)


def test_kettle_turn_on_oracle_true_when_knob_in_range() -> None:
    rc = _kettle_robocasa(_kettle_kitchen(burner_sites=_BURNER_SITES_OK))
    assert rc._kettle_turn_on_oracle(_KETTLE_TURN_STEP) is True


def test_kettle_turn_on_oracle_false_when_knob_at_zero() -> None:
    kitchen = _kettle_kitchen(
        burner_sites=_BURNER_SITES_OK, knobs={"rear_right": 0.0})
    rc = _kettle_robocasa(kitchen)
    assert rc._kettle_turn_on_oracle(_KETTLE_TURN_STEP) is False


def test_kettle_turn_on_oracle_bad_burner_phrase_raises() -> None:
    rc = _kettle_robocasa(_kettle_kitchen(burner_sites=_BURNER_SITES_OK))
    step = {"skill": "turn_on", "target_name": "left drawer"}
    with pytest.raises(ValueError, match="burner phrase"):
        rc._kettle_turn_on_oracle(step)


def test_kettle_check_subtask_open_raises_key_error() -> None:
    rc = _kettle_robocasa(_kettle_kitchen(burner_sites=_BURNER_SITES_OK))
    with pytest.raises(KeyError, match="unsupported skill"):
        rc.check_subtask({"skill": "open", "object_name": "right drawer"})


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
