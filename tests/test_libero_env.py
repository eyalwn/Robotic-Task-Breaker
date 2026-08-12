"""Hermetic tests for LiberoEnv init-state and subtask-check helpers."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import libero_env


def _make_obs() -> dict:
    return {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def _build_env_with_mocks(
    *,
    init_states: list[object],
    init_state_idx: int = 0,
    goal_state: list[list[str]] | None = None,
) -> libero_env.LiberoEnv:
    goal_state = goal_state or [
        ["in", "alphabet_soup_1", "basket_1_contain_region"],
        ["in", "tomato_sauce_1", "basket_1_contain_region"],
    ]
    inner = MagicMock()
    inner.parsed_problem = {"goal_state": goal_state}
    inner.object_states_dict = {
        "alphabet_soup_1": SimpleNamespace(name="alphabet_soup_1"),
        "tomato_sauce_1": SimpleNamespace(name="tomato_sauce_1"),
        "basket_1_contain_region": SimpleNamespace(name="basket_1_contain_region"),
        "microwave_1": SimpleNamespace(name="microwave_1"),
        "stove_1": SimpleNamespace(name="stove_1"),
        "plate_1": SimpleNamespace(name="plate_1"),
        "table_1": SimpleNamespace(name="table_1"),
    }
    gripper = SimpleNamespace(
        important_geoms={
            "left_fingerpad": ["left_pad_geom"],
            "right_fingerpad": ["right_pad_geom"],
        },
        contact_geoms=["left_pad_geom", "right_pad_geom"],
    )
    inner.robots = [SimpleNamespace(gripper=gripper)]
    inner.get_object.return_value = SimpleNamespace(contact_geoms=["geom_a"])
    inner.check_contact.return_value = True

    offscreen = MagicMock()
    offscreen.env = inner
    offscreen.set_init_state.return_value = _make_obs()
    offscreen.step.return_value = (_make_obs(), 0.0, False, {})
    offscreen.check_success.return_value = False
    offscreen_cls = MagicMock(return_value=offscreen)

    task = SimpleNamespace(
        language="put both the alphabet soup and the tomato sauce in the basket",
        problem_folder="libero_10",
        bddl_file="task0.bddl",
    )
    suite = MagicMock()
    suite.get_task.return_value = task
    suite.get_task_init_states.return_value = init_states

    benchmark = MagicMock()
    benchmark.get_benchmark_dict.return_value = {"libero_10": lambda: suite}
    get_predicate_fn = MagicMock(
        side_effect=lambda name: (
            lambda obj1, obj2=None: (
                obj1.name.startswith("alphabet")
                if name in {"in", "on"}
                else name in {"close", "turnon"}
            )
        )
    )

    with patch.object(
        libero_env,
        "_import_libero",
        return_value=(benchmark, lambda _: "/bddl", offscreen_cls, get_predicate_fn),
    ):
        return libero_env.LiberoEnv("libero_10", 0, init_state_idx=init_state_idx)


def test_init_state_idx_for_episode_wraps():
    assert libero_env.LiberoEnv.init_state_idx_for_episode(0, 5) == 0
    assert libero_env.LiberoEnv.init_state_idx_for_episode(7, 5) == 2


def test_init_state_idx_for_episode_rejects_invalid():
    with pytest.raises(ValueError, match="init_state_count"):
        libero_env.LiberoEnv.init_state_idx_for_episode(0, 0)
    with pytest.raises(ValueError, match="episode_idx"):
        libero_env.LiberoEnv.init_state_idx_for_episode(-1, 3)


def test_init_state_idx_out_of_range():
    with pytest.raises(ValueError, match="init_state_idx 3 out of range"):
        _build_env_with_mocks(init_states=[b"a", b"b", b"c"], init_state_idx=3)


def test_init_state_idx_selects_requested_state():
    init_states = [b"state0", b"state1"]
    env = _build_env_with_mocks(init_states=init_states, init_state_idx=1)
    try:
        assert env.init_state_idx == 1
        assert env.init_state_count() == 2
        env._env.set_init_state.assert_called_once_with(b"state1")
    finally:
        env.close()


def test_goal_conjuncts_returns_copy():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        conjuncts = env.goal_conjuncts()
        assert conjuncts == [
            ["in", "alphabet_soup_1", "basket_1_contain_region"],
            ["in", "tomato_sauce_1", "basket_1_contain_region"],
        ]
        conjuncts[0][0] = "mutated"
        assert env.goal_conjuncts()[0][0] == "in"
    finally:
        env.close()


def test_eval_goal_conjunct_uses_inner_predicate():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        assert env.eval_goal_conjunct(0) is True
        assert env.eval_goal_conjunct(1) is False
        with pytest.raises(ValueError, match="goal conjunct index 9"):
            env.eval_goal_conjunct(9)
    finally:
        env.close()


def test_eval_predicate_builds_lowercase_tuple():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        assert env.eval_predicate("In", "alphabet_soup_1", "basket_1_contain_region")
        with pytest.raises(ValueError, match="predicate needs"):
            env.eval_predicate("on", "obj", "region", "extra")
    finally:
        env.close()


def test_is_grasping_uses_robosuite_check():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        assert env.is_grasping("alphabet_soup_1") is True
        inner = env._get_sim_gt()
        inner.get_object.assert_called_once_with("alphabet_soup_1")
        assert inner.check_contact.call_count == 2
        inner.check_contact.assert_any_call(["left_pad_geom"], ["geom_a"])
        inner.check_contact.assert_any_call(["right_pad_geom"], ["geom_a"])
    finally:
        env.close()


def test_check_subtask_grab_uses_is_grasping():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        step = {"skill": "grab", "object_name": "alphabet_soup_1"}
        assert env.check_subtask(step) is True
        env._get_sim_gt().get_object.assert_called_with("alphabet_soup_1")
    finally:
        env.close()


def test_check_subtask_place_inside_uses_in_predicate():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        step = {
            "skill": "place",
            "object_name": "alphabet_soup_1",
            "target_name": "basket_1_contain_region",
            "position": "inside",
        }
        assert env.check_subtask(step) is True
    finally:
        env.close()


def test_check_subtask_place_on_uses_on_predicate():
    env = _build_env_with_mocks(init_states=[b"s"])
    env._get_predicate_fn = lambda name: (
        lambda obj1, obj2=None: name == "on" and obj1.name.startswith("plate")
    )
    try:
        step = {
            "skill": "place",
            "object_name": "plate_1",
            "target_name": "table_1",
            "position": "on",
        }
        assert env.check_subtask(step) is True
    finally:
        env.close()


def test_check_subtask_close_and_turn_on():
    env = _build_env_with_mocks(init_states=[b"s"])
    env._get_predicate_fn = lambda name: (lambda obj1, obj2=None: name in {"close", "turnon"})
    try:
        assert env.check_subtask({"skill": "close", "target_name": "microwave_1"})
        assert env.check_subtask({"skill": "turn_on", "target_name": "stove_1"})
    finally:
        env.close()


def test_check_subtask_grab_resolves_nl_name():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        step = {"skill": "grab", "object_name": "alphabet soup"}
        assert env.check_subtask(step) is True
        env._get_sim_gt().get_object.assert_called_with("alphabet_soup_1")
    finally:
        env.close()


def test_check_subtask_place_inside_resolves_nl_to_contain_region():
    env = _build_env_with_mocks(init_states=[b"s"])
    inner = env._get_sim_gt()
    inner.object_states_dict["basket_1"] = SimpleNamespace(name="basket_1")
    try:
        with patch.object(env, "eval_predicate", return_value=True) as eval_pred:
            step = {
                "skill": "place",
                "object_name": "alphabet soup",
                "target_name": "basket",
                "position": "inside",
            }
            assert env.check_subtask(step) is True
            eval_pred.assert_called_once_with(
                "in", "alphabet_soup_1", "basket_1_contain_region"
            )
    finally:
        env.close()


def test_resolve_name_accepts_exact_gt_id():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        assert env._resolve_name("alphabet_soup_1") == "alphabet_soup_1"
        assert (
            env._resolve_name("basket_1_contain_region", prefer_region=True)
            == "basket_1_contain_region"
        )
    finally:
        env.close()


def test_resolve_name_unknown_raises():
    env = _build_env_with_mocks(init_states=[b"s"])
    try:
        with pytest.raises(ValueError, match="unknown object name 'no such object'"):
            env._resolve_name("no such object")
    finally:
        env.close()


def test_resolve_name_ambiguous_raises():
    env = _build_env_with_mocks(init_states=[b"s"])
    inner = env._get_sim_gt()
    inner.object_states_dict["alphabet_soup_2"] = SimpleNamespace(name="alphabet_soup_2")
    try:
        with pytest.raises(ValueError, match="ambiguous object name 'alphabet soup'"):
            env._resolve_name("alphabet soup")
    finally:
        env.close()


def test_is_grasping_unknown_object_raises():
    env = _build_env_with_mocks(init_states=[b"s"])
    env._get_sim_gt().get_object.return_value = None
    try:
        with pytest.raises(ValueError, match="unknown object for grasp check"):
            env.is_grasping("alphabet_soup_1")
    finally:
        env.close()
