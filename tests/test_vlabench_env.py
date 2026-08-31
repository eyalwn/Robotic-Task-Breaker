"""Hermetic tests for VLABenchEnv observation, stepping, and subtask checks."""
from __future__ import annotations

import builtins
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import envs.vlabench_env as vlabench_env


def _make_rgb() -> np.ndarray:
    return np.zeros((3, 8, 8, 3), dtype=np.uint8)


def _make_obs() -> dict:
    return {
        "rgb": _make_rgb(),
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


def test_init_calls_load_env_once():
    env = _build_env_with_mocks()
    try:
        assert env._mock_load_env.call_count == 1
        assert env.task_description == "pick the apple"
    finally:
        env.close()


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


def test_check_subtask_grab_and_place():
    env = _build_env_with_mocks()
    try:
        assert env.check_subtask({"skill": "grab", "object_name": "apple"}) is True
        place_step = {
            "skill": "place",
            "object_name": "apple",
            "target_name": "basket",
            "position": "inside",
        }
        assert env.check_subtask(place_step) is True
        with pytest.raises(AssertionError, match="close"):
            env.check_subtask({"skill": "close", "target_name": "basket"})
    finally:
        env.close()


def test_place_target_without_contain_raises():
    entities = {
        "apple": SimpleNamespace(get_xpos=lambda physics: np.zeros(3)),
        "table": SimpleNamespace(get_xpos=lambda physics: np.zeros(3)),
    }
    env = _build_env_with_mocks(entities=entities)
    try:
        step = {
            "skill": "place",
            "object_name": "apple",
            "target_name": "table",
            "position": "on",
        }
        with pytest.raises(ValueError, match="is not a receptacle"):
            env.check_subtask(step)
    finally:
        env.close()
