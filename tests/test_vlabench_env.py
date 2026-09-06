"""Hermetic tests for VLABenchEnv observation, stepping, and subtask checks."""
from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import envs.vlabench_env as vlabench_env


def _make_rgb(*, markers: tuple[int, int, int, int] | None = None) -> np.ndarray:
    markers = markers or (0, 0, 0, 0)
    frames = []
    for value in markers:
        frames.append(np.full((8, 8, 3), value, dtype=np.uint8))
    return np.stack(frames, axis=0)


def _make_obs(*, rgb_markers: tuple[int, int, int, int] = (10, 20, 30, 40)) -> dict:
    return {
        "rgb": _make_rgb(markers=rgb_markers),
        "ee_state": np.array(
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5],
            dtype=np.float64,
        ),
    }


def _build_env_with_mocks(
  *,
  entities: dict | None = None,
  robot_frame_position: object = np.array([0.0, -0.4, 0.78], dtype=np.float64),
) -> vlabench_env.VLABenchEnv:
    entities = entities or {
        "apple": SimpleNamespace(
            is_grasped=lambda physics, robot: True,
            get_xpos=lambda physics: np.array([0.0, 0.0, 0.0]),
        ),
        "alphabet_soup": SimpleNamespace(
            is_grasped=lambda physics, robot: False,
            get_xpos=lambda physics: np.array([1.0, 0.0, 0.0]),
        ),
        "basket": SimpleNamespace(
            contain=lambda obj_pos, physics: True,
            get_xpos=lambda physics: np.array([2.0, 0.0, 0.0]),
        ),
    }
    task = SimpleNamespace(
        get_instruction=lambda: "pick the apple",
        entities=entities,
        robot=SimpleNamespace(
            n_dof=7,
            end_effector_site=SimpleNamespace(full_identifier="ee_site"),
        ),
        should_terminate_episode=lambda physics: False,
    )
    physics = MagicMock()
    physics.data.ctrl.shape = (8,)
    env = MagicMock()
    env.task = task
    env.physics = physics
    env.get_robot_frame_position.return_value = robot_frame_position
    env.get_observation.return_value = _make_obs()
    env.step.return_value = None
    env.close.return_value = None

    load_env = MagicMock(return_value=env)

    with patch.object(
        vlabench_env,
        "_import_vlabench",
        return_value=load_env,
    ):
        venv = vlabench_env.VLABenchEnv("select_fruit")
    venv._mock_load_env = load_env
    venv._mock_env = env
    return venv


def _patch_dm_control_ik(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    ik_mock = MagicMock(return_value=SimpleNamespace(qpos=np.zeros(7)))
    ik_mod = types.ModuleType("dm_control.utils.inverse_kinematics")
    ik_mod.qpos_from_site_pose = ik_mock
    utils_mod = types.ModuleType("dm_control.utils")
    utils_mod.inverse_kinematics = ik_mod
    dm_mod = types.ModuleType("dm_control")
    dm_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "dm_control", dm_mod)
    monkeypatch.setitem(sys.modules, "dm_control.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "dm_control.utils.inverse_kinematics", ik_mod)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dm_control.utils.inverse_kinematics":
            return ik_mod
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return ik_mock


def _install_fake_vlabench_package(
    monkeypatch: pytest.MonkeyPatch,
    package_root: Path,
) -> None:
    package_root.mkdir(parents=True, exist_ok=True)
    init_file = package_root / "__init__.py"
    init_file.write_text("# fake VLABench package\n", encoding="utf-8")
    module = types.ModuleType("VLABench")
    module.__file__ = str(init_file)
    monkeypatch.setitem(sys.modules, "VLABench", module)


def test_observe_returns_front_second_wrist_order_and_7d_state():
    """rgb[2,0,3] -> front/second/wrist JPEGs; ee_state -> robot-frame 7-D state."""
    robot_base = np.array([0.0, -0.4, 0.78], dtype=np.float64)
    obs = {
        "rgb": _make_rgb(markers=(11, 22, 33, 44)),
        "ee_state": np.array(
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5],
            dtype=np.float64,
        ),
    }
    load_env = MagicMock()
    load_env.return_value = MagicMock(
        task=SimpleNamespace(get_instruction=lambda: "pick fruit"),
        get_robot_frame_position=MagicMock(return_value=robot_base),
        get_observation=MagicMock(return_value=obs),
        close=MagicMock(),
    )
    with patch.object(vlabench_env, "_import_vlabench", return_value=load_env), patch.object(
        vlabench_env,
        "to_jpeg",
        side_effect=lambda rgb: bytes([int(rgb[0, 0, 0])]),
    ):
        env = vlabench_env.VLABenchEnv("select_fruit")
        try:
            images, state = env.observe()
        finally:
            env.close()

    assert images == (bytes([33]), bytes([11]), bytes([44]))
    assert len(state) == 7
    assert state[0] == pytest.approx(0.1)
    assert state[1] == pytest.approx(0.6)
    assert state[2] == pytest.approx(-0.48)
    assert state[3:6] == pytest.approx([0.0, 0.0, 0.0])
    assert state[6] == pytest.approx(0.5)


def test_get_track1_episode_configs_derives_path_from_installed_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "site-packages" / "VLABench"
    _install_fake_vlabench_package(monkeypatch, package_root)
    track_dir = package_root / "configs" / "evaluation" / "tracks"
    track_dir.mkdir(parents=True)
    payload = {
        "select_fruit": [
            {"instructions": ["episode-0"]},
            {"instructions": ["episode-1"]},
        ]
    }
    track_file = track_dir / "track_1_in_distribution.json"
    track_file.write_text(json.dumps(payload), encoding="utf-8")

    configs = vlabench_env.get_track1_episode_configs("select_fruit")

    assert configs == payload["select_fruit"]


def test_get_track1_episode_configs_raises_for_missing_track_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "site-packages" / "VLABench"
    _install_fake_vlabench_package(monkeypatch, package_root)

    with pytest.raises(RuntimeError, match="Track-1 file not found"):
        vlabench_env.get_track1_episode_configs("select_fruit")


def test_get_track1_episode_configs_raises_for_missing_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "site-packages" / "VLABench"
    _install_fake_vlabench_package(monkeypatch, package_root)
    track_dir = package_root / "configs" / "evaluation" / "tracks"
    track_dir.mkdir(parents=True)
    payload = {"other_task": [{"instructions": ["episode-0"]}]}
    track_file = track_dir / "track_1_in_distribution.json"
    track_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="has no episodes for task 'select_fruit'"):
        vlabench_env.get_track1_episode_configs("select_fruit")


def test_get_track1_episode_config_raises_runtime_error_for_out_of_range_index() -> None:
    payload = [{"instructions": ["episode-0"]}]
    with patch.object(vlabench_env, "get_track1_episode_configs", return_value=payload):
        with pytest.raises(
            RuntimeError,
            match="task_idx=1.*task='select_fruit'.*available_config_count=1",
        ):
            vlabench_env.get_track1_episode_config("select_fruit", 1)


def test_from_track_episode_index_propagates_invalid_index_error() -> None:
    with patch.object(
        vlabench_env,
        "get_track1_episode_config",
        side_effect=RuntimeError(
            "Invalid Track-1 task_idx=9 for task='select_fruit'; available_config_count=2"
        ),
    ), patch.object(vlabench_env.VLABenchEnv, "from_track_episode") as from_episode:
        with pytest.raises(
            RuntimeError,
            match="task_idx=9.*task='select_fruit'.*available_config_count=2",
        ):
            vlabench_env.VLABenchEnv.from_track_episode_index("select_fruit", 9)
    from_episode.assert_not_called()


def test_init_calls_load_env_once():
    env = _build_env_with_mocks()
    try:
        assert env._mock_load_env.call_count == 1
        env._mock_load_env.assert_called_once_with(
            task="select_fruit",
            render_resolution=(480, 480),
        )
        assert env.task_description == "pick the apple"
    finally:
        env.close()


def test_from_track_episode_uses_eval_kwargs():
    load_env = MagicMock()
    load_env.return_value = MagicMock(
        task=SimpleNamespace(get_instruction=lambda: "pick fruit"),
        get_robot_frame_position=MagicMock(return_value=np.array([0.0, -0.4, 0.78])),
        close=MagicMock(),
    )
    episode_config = {"instructions": ["put fruit on plate"]}
    with patch.object(vlabench_env, "_import_vlabench", return_value=load_env):
        env = vlabench_env.VLABenchEnv.from_track_episode("select_fruit", episode_config)
    try:
        load_env.assert_called_once_with(
            task="select_fruit",
            render_resolution=(480, 480),
            episode_config=episode_config,
            random_init=False,
            run_mode="eval",
        )
    finally:
        env.close()


def test_from_track_episode_index_builds_selected_config():
    episode_config = {"instructions": ["episode-2"]}
    with patch.object(
        vlabench_env,
        "get_track1_episode_config",
        return_value=episode_config,
    ) as get_cfg, patch.object(
        vlabench_env.VLABenchEnv,
        "from_track_episode",
        return_value="env",
    ) as from_cfg:
        env = vlabench_env.VLABenchEnv.from_track_episode_index("select_fruit", 2)

    get_cfg.assert_called_once_with("select_fruit", 2)
    from_cfg.assert_called_once_with("select_fruit", episode_config)
    assert env == "env"


def test_init_raises_if_instruction_missing():
    load_env = MagicMock()
    with patch.object(
        vlabench_env,
        "_import_vlabench",
        return_value=load_env,
    ):
        load_env.return_value = MagicMock(
            task=SimpleNamespace(get_instruction=lambda: None),
        )
        with pytest.raises(RuntimeError, match="no language instruction"):
            vlabench_env.VLABenchEnv("select_fruit")


def test_init_raises_if_robot_frame_getter_fails():
    load_env = MagicMock()
    with patch.object(
        vlabench_env,
        "_import_vlabench",
        return_value=load_env,
    ):
        load_env.return_value = MagicMock(
            task=SimpleNamespace(get_instruction=lambda: "task"),
            get_robot_frame_position=MagicMock(side_effect=RuntimeError("no frame")),
        )
        with pytest.raises(RuntimeError, match="no frame"):
            vlabench_env.VLABenchEnv("select_fruit")


def test_step_requires_7d_action(monkeypatch: pytest.MonkeyPatch):
    env = _build_env_with_mocks()
    try:
        ik = _patch_dm_control_ik(monkeypatch)
        assert env.step(np.zeros(7)) is False
        env._mock_env.step.assert_called_once()
        ik.assert_called_once()
        with pytest.raises(ValueError, match="expected 7-D EEF action"):
            env.step(np.zeros(8))
        assert env._mock_env.step.call_count == 1
    finally:
        env.close()


def test_step_does_not_passthrough_joint_ctrl(monkeypatch: pytest.MonkeyPatch):
    env = _build_env_with_mocks()
    try:
        _patch_dm_control_ik(monkeypatch)
        with pytest.raises(ValueError, match="expected 7-D EEF action"):
            env.step(np.zeros(8))
        env._mock_env.step.assert_not_called()
    finally:
        env.close()


def test_step_binarizes_gripper(monkeypatch: pytest.MonkeyPatch):
    env = _build_env_with_mocks()
    try:
        _patch_dm_control_ik(monkeypatch)
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20]))
        ctrl_open = env._mock_env.step.call_args.args[0]
        assert ctrl_open[-1] == pytest.approx(0.04)
        env._mock_env.step.reset_mock()
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]))
        ctrl_closed = env._mock_env.step.call_args.args[0]
        assert ctrl_closed[-1] == pytest.approx(0.0)
    finally:
        env.close()


def test_resolve_entity_exact_and_normalized():
    env = _build_env_with_mocks()
    try:
        assert env._resolve_entity("apple") is env._mock_env.task.entities["apple"]
        assert (
            env._resolve_entity("Alphabet Soup")
            is env._mock_env.task.entities["alphabet_soup"]
        )
        with pytest.raises(ValueError, match="unknown object name 'orange'"):
            env._resolve_entity("orange")
    finally:
        env.close()


def test_check_subtask_transfer_uses_contain():
    env = _build_env_with_mocks()
    try:
        transfer_step = {
            "skill": "transfer",
            "object_name": "apple",
            "target_name": "basket",
        }
        assert env.check_subtask(transfer_step) is True
        with pytest.raises(KeyError, match="unsupported skill"):
            env.check_subtask({"skill": "close", "target_name": "basket"})
    finally:
        env.close()


def test_check_subtask_transfer_false_when_contain_false():
    entities = {
        "apple": SimpleNamespace(get_xpos=lambda physics: np.zeros(3)),
        "basket": SimpleNamespace(
            contain=lambda obj_pos, physics: False,
            get_xpos=lambda physics: np.zeros(3),
        ),
    }
    env = _build_env_with_mocks(entities=entities)
    try:
        step = {
            "skill": "transfer",
            "object_name": "apple",
            "target_name": "basket",
        }
        assert env.check_subtask(step) is False
    finally:
        env.close()


def test_transfer_target_without_contain_raises():
    entities = {
        "apple": SimpleNamespace(get_xpos=lambda physics: np.zeros(3)),
        "table": SimpleNamespace(get_xpos=lambda physics: np.zeros(3)),
    }
    env = _build_env_with_mocks(entities=entities)
    try:
        step = {
            "skill": "transfer",
            "object_name": "apple",
            "target_name": "table",
        }
        with pytest.raises(ValueError, match="is not a receptacle"):
            env.check_subtask(step)
    finally:
        env.close()


_EXPECTED_COOK_DISHES_EPISODE_SHA256 = [
    "2a2e33d510f806cecfab995b61814ad81ace11eda349d695e4a650c41789f3df",
    "b8ffc62e1ad0cf5f3d543576f219d852fdb68a86d2925500a746afc48026f671",
    "a5f8425154d32569572eeabbf4365cddc3ec6e8e5b602d95934132b323e9fdc5",
    "473c8b2503f58605b4537c84b4a8f4bf896e62fb29390f8a6391c36590106541",
    "151745be97f8f522b3b127e882c1a940338705f612545b707840a7d589f4f498",
    "a8fdfee7f75ec8251048ac1c47cbc6d9e716de818a1eeb76e62520a3b2206983",
    "afb577525cc6d3070c15d6f1b1952818332945ca98e7e14bfe8e1b248c6ee163",
    "3e2440775a47250562a6bf88a1f994d13f33878195b5409640f1a0a6fda140b2",
    "868b3f2ef60363b14b7c02ae09427050b3f1202becaed9190d293b87b4923d73",
    "1532c225143401a9bffaea9966b69b35c2245d3faf04ab9be22cb5d6999e0421",
]


def test_cook_dishes_evaluation_data_provenance() -> None:
    eval_file = Path(vlabench_env.__file__).with_name(
        "cook_dishes_evaluation_episodes.json"
    )
    with eval_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    provenance = payload["provenance"]
    assert provenance["dataset"] == "VLABench/vlm_evaluation_v1.0"
    assert provenance["revision"] == "b432f0c69ec0731935ff49a916d460936a000624"
    assert provenance["task"] == "cook_dishes"
    assert provenance["path_template"] == (
        "Complex/cook_dishes/example{N}/env_config/env_config.json"
    )
    assert provenance["episode_sha256"] == _EXPECTED_COOK_DISHES_EPISODE_SHA256


def test_cook_dishes_evaluation_episode_sha256_integrity() -> None:
    eval_file = Path(vlabench_env.__file__).with_name(
        "cook_dishes_evaluation_episodes.json"
    )
    with eval_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload["episodes"]
    digests = [
        vlabench_env._episode_sha256(dict(episode)) for episode in episodes
    ]
    assert digests == _EXPECTED_COOK_DISHES_EPISODE_SHA256
    configs = vlabench_env.get_evaluation_episode_configs("cook_dishes")
    loaded_digests = [vlabench_env._episode_sha256(config) for config in configs]
    assert loaded_digests == _EXPECTED_COOK_DISHES_EPISODE_SHA256


def test_cook_dishes_evaluation_has_ten_episodes() -> None:
    configs = vlabench_env.get_evaluation_episode_configs("cook_dishes")
    assert len(configs) == 10
    for config in configs:
        assert isinstance(config, dict)
        assert "task" in config


def test_get_evaluation_episode_config_cook_dishes_bounds() -> None:
    with pytest.raises(
        RuntimeError,
        match="task_idx=10.*task='cook_dishes'.*available_config_count=10",
    ):
        vlabench_env.get_evaluation_episode_config("cook_dishes", 10)


def test_get_evaluation_episode_configs_primitive_delegates_to_track1() -> None:
    payload = [{"instructions": ["episode-0"]}]
    with patch.object(
        vlabench_env,
        "get_track1_episode_configs",
        return_value=payload,
    ) as track1:
        configs = vlabench_env.get_evaluation_episode_configs("select_fruit")
    track1.assert_called_once_with("select_fruit")
    assert configs == payload


def test_validate_evaluation_episode_range_rejects_negative_start() -> None:
    with pytest.raises(RuntimeError, match="task_idx=-1.*must be >= 0"):
        vlabench_env.validate_evaluation_episode_range("cook_dishes", -1, 1)


def test_validate_evaluation_episode_range_rejects_out_of_range_end() -> None:
    with pytest.raises(
        RuntimeError,
        match="task_idx=8 episodes=3.*available_config_count=10",
    ):
        vlabench_env.validate_evaluation_episode_range("cook_dishes", 8, 3)


def test_validate_evaluation_episode_range_accepts_full_campaign() -> None:
    vlabench_env.validate_evaluation_episode_range("cook_dishes", 0, 10)


def test_from_evaluation_episode_index_cook_dishes_uses_static_config() -> None:
    configs = vlabench_env.get_evaluation_episode_configs("cook_dishes")
    with patch.object(
        vlabench_env.VLABenchEnv,
        "from_track_episode",
        return_value="env",
    ) as from_episode:
        env = vlabench_env.VLABenchEnv.from_evaluation_episode_index("cook_dishes", 3)
    from_episode.assert_called_once_with("cook_dishes", configs[3])
    assert env == "env"


def test_from_evaluation_episode_index_primitive_uses_track1() -> None:
    episode_config = {"instructions": ["episode-2"]}
    with patch.object(
        vlabench_env,
        "get_track1_episode_configs",
        return_value=[{}, {}, episode_config],
    ), patch.object(
        vlabench_env.VLABenchEnv,
        "from_track_episode",
        return_value="env",
    ) as from_episode:
        env = vlabench_env.VLABenchEnv.from_evaluation_episode_index("select_fruit", 2)
    from_episode.assert_called_once_with("select_fruit", episode_config)
    assert env == "env"
