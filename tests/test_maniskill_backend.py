"""Tests for sim.maniskill_backend lazy-import behaviour and drive/camera contracts."""
import sys
import types

import numpy as np
import pytest

from conftest import FakeEnv


def test_import_without_maniskill_does_not_crash():
	"""The module must be importable even when mani_skill is absent."""
	import importlib
	import sys

	# Temporarily hide mani_skill and gymnasium from the import machinery
	hidden = {}
	for pkg in ("mani_skill", "gymnasium"):
		if pkg in sys.modules:
			hidden[pkg] = sys.modules.pop(pkg)
		sys.modules[pkg] = None  # type: ignore[assignment]  # makes import fail

	# Also remove cached module so it re-runs module-level code
	sys.modules.pop("sim.maniskill_backend", None)

	try:
		import sim.maniskill_backend  # must not raise
	finally:
		# Restore
		for pkg, mod in hidden.items():
			sys.modules[pkg] = mod
		for pkg in ("mani_skill", "gymnasium"):
			if pkg not in hidden:
				sys.modules.pop(pkg, None)
		sys.modules.pop("sim.maniskill_backend", None)


def test_create_env_raises_without_maniskill(monkeypatch):
	"""create_env() must raise RuntimeError with a helpful message when deps missing."""
	import sys
	from sim import maniskill_backend

	hidden = {}
	for pkg in ("mani_skill", "gymnasium"):
		if pkg in sys.modules:
			hidden[pkg] = sys.modules.pop(pkg)
		sys.modules[pkg] = None  # type: ignore[assignment]

	try:
		with pytest.raises(RuntimeError, match="ManiSkill"):
			maniskill_backend.create_env(env_id="PickCube-v1")
	finally:
		for pkg, mod in hidden.items():
			sys.modules[pkg] = mod
		for pkg in ("mani_skill", "gymnasium"):
			if pkg not in hidden:
				sys.modules.pop(pkg, None)


def _install_fake_runtime(monkeypatch, make_impl):
    fake_gym = types.ModuleType("gymnasium")
    fake_gym.make = make_impl
    fake_gym.envs = types.SimpleNamespace(registry={})
    fake_mani = types.ModuleType("mani_skill")
    fake_mani_envs = types.ModuleType("mani_skill.envs")
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    monkeypatch.setitem(sys.modules, "mani_skill", fake_mani)
    monkeypatch.setitem(sys.modules, "mani_skill.envs", fake_mani_envs)


class _FakeEnv:
    def __init__(self):
        self.reset_calls = 0
        self.unwrapped = self

    def reset(self, **_kwargs):
        self.reset_calls += 1
        return None, {}


def test_create_env_retries_rgbd_before_success(monkeypatch):
    from sim import maniskill_backend

    calls = {"n": 0}

    def _make(_env_id, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("vk::Queue::submit: ErrorDeviceLost")
        env = _FakeEnv()
        env._kwargs = kwargs
        return env

    _install_fake_runtime(monkeypatch, _make)
    monkeypatch.setenv("MANISKILL_GYM_MAKE_RETRIES", "3")
    monkeypatch.setenv("ROBOCREW_REQUIRE_RGBD", "1")

    env = maniskill_backend.create_env(env_id="PickCube-v1")
    assert calls["n"] == 3
    assert getattr(env, "_sim_obs_mode") == "rgbd"


def test_create_env_requires_rgbd_raises_instead_of_state_fallback(monkeypatch):
    from sim import maniskill_backend

    def _make(_env_id, **_kwargs):
        raise RuntimeError("vk::Queue::submit: ErrorDeviceLost")

    _install_fake_runtime(monkeypatch, _make)
    monkeypatch.setenv("MANISKILL_GYM_MAKE_RETRIES", "2")
    monkeypatch.setenv("ROBOCREW_REQUIRE_RGBD", "1")

    with pytest.raises(RuntimeError, match="rgbd observation is required"):
        maniskill_backend.create_env(env_id="PickCube-v1")


def test_create_env_forwards_sim_render_and_obs_overrides(monkeypatch):
    from sim import maniskill_backend

    seen = {}

    def _make(_env_id, **kwargs):
        seen.update(kwargs)
        return _FakeEnv()

    _install_fake_runtime(monkeypatch, _make)
    monkeypatch.setenv("MANISKILL_RENDER_BACKEND", "sapien_cpu")
    monkeypatch.setenv("MANISKILL_SIM_BACKEND", "physx_cpu")
    monkeypatch.setenv("MANISKILL_OBS_MODE", "state")

    env = maniskill_backend.create_env(env_id="PickCube-v1", obs_mode="rgbd")
    assert seen["render_backend"] == "sapien_cpu"
    assert seen["sim_backend"] == "physx_cpu"
    assert seen["obs_mode"] == "state"
    assert getattr(env, "_sim_obs_mode") == "state"


# ---------------------------------------------------------------------------
# ManiSkillDrive / ManiSkillCamera contract tests (FakeEnv)
# ---------------------------------------------------------------------------

from sim.maniskill_backend import (
    ManiSkillCamera,
    ManiSkillDrive,
    _STEPS_PER_DEGREE,
    _STEPS_PER_METER,
)


def _make_drive():
    env = FakeEnv()
    drive = ManiSkillDrive(env)
    return env, drive


def test_move_forward_step_count():
    distance = 0.5
    env, drive = _make_drive()
    drive.move_linear(distance)
    expected = max(1, int(abs(distance) * _STEPS_PER_METER))
    assert len(env.actions) == expected


def test_move_backward_step_count():
    distance = -0.5
    env, drive = _make_drive()
    drive.move_linear(distance)
    expected = max(1, int(abs(distance) * _STEPS_PER_METER))
    assert len(env.actions) == expected


def test_move_linear_sign():
    env_fwd, drive_fwd = _make_drive()
    drive_fwd.move_linear(0.5)
    env_bwd, drive_bwd = _make_drive()
    drive_bwd.move_linear(-0.5)

    fwd_linear = env_fwd.actions[0][0]
    bwd_linear = env_bwd.actions[0][0]
    assert fwd_linear > 0, "forward should produce positive linear action"
    assert bwd_linear < 0, "backward should produce negative linear action"


def test_turn_in_place_step_count():
    angle = 30.0
    env, drive = _make_drive()
    drive.turn_in_place(angle)
    expected = max(1, int(abs(angle) * _STEPS_PER_DEGREE))
    assert len(env.actions) == expected


def test_turn_left_sign():
    """Positive angle → positive angular action dim (left/CCW)."""
    env, drive = _make_drive()
    drive.turn_in_place(30.0)
    assert env.actions[0][1] > 0, "positive angle should produce positive angular action"


def test_turn_right_sign():
    """Negative angle → negative angular action dim (right/CW)."""
    env, drive = _make_drive()
    drive.turn_in_place(-30.0)
    assert env.actions[0][1] < 0, "negative angle should produce negative angular action"


def test_capture_image_returns_jpeg():
    import cv2

    env = FakeEnv()
    cam = ManiSkillCamera(env)
    data = cam.capture_image()

    assert isinstance(data, bytes)
    assert len(data) > 0

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None, "JPEG bytes must decode to a valid image"
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_maniskill_backend_apply_env_defaults(monkeypatch):
    import os

    from sim.maniskill_backend import ManiSkillBackend

    for key in (
        "VLA_AXIS_REMAP",
        "VLA_AXIS_SIGNS",
        "VLA_SIM_PICKCUBE_GRIPPER_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)
    ManiSkillBackend._apply_env_defaults()
    assert os.environ.get("VLA_AXIS_REMAP") == "2,0,1"
    assert os.environ.get("VLA_AXIS_SIGNS") == "1,-1,-1"
    assert os.environ.get("VLA_SIM_PICKCUBE_GRIPPER_PROFILE") == "true"


def test_maniskill_backend_register_skills(monkeypatch):
    from core.skills import SkillRegistry
    from sim.maniskill_backend import ManiSkillBackend

    env = FakeEnv()
    arm = object()
    camera = object()
    checker = type("Checker", (), {"check": lambda self: False})()

    backend = ManiSkillBackend.__new__(ManiSkillBackend)
    backend._env = env
    backend._arm = arm
    backend._camera = camera
    backend._checker = checker
    backend._vla_port = 8765

    goal = "pick the cube"
    vla_kwargs: dict = {}
    arm_skill_calls: list[dict] = []

    class _StubTool:
        def invoke(self, args):
            return "ok"

    def _record_vla(**kwargs):
        vla_kwargs.update(kwargs)
        return _StubTool()

    def _record_arm_skills(registry, vla_callable, arm=None):
        arm_skill_calls.append({"arm": arm})
        return _orig_register_arm_skills(registry, vla_callable, arm=arm)

    import vla.arm_skills as arm_skills_mod

    _orig_register_arm_skills = arm_skills_mod.register_arm_skills
    monkeypatch.setattr("vla.create_vla_manipulation", _record_vla)
    monkeypatch.setattr("vla.VLAConfig.from_env", lambda: type("Cfg", (), {})())
    monkeypatch.setattr(arm_skills_mod, "register_arm_skills", _record_arm_skills)

    registry = SkillRegistry()
    backend.register_skills(registry, goal)

    names = {s.name for s in registry._skills.values()}
    assert "state.list_objects" in names
    assert "perception.locate" in names
    assert "arm.grab" in names
    assert "arm.place" in names
    assert vla_kwargs.get("task_prompt") == goal
    assert len(arm_skill_calls) == 1
    assert arm_skill_calls[0]["arm"] is None


def test_maniskill_backend_success_delegates_to_checker():
    from sim.maniskill_backend import ManiSkillBackend

    class _Checker:
        def __init__(self):
            self.calls = 0

        def check(self):
            self.calls += 1
            return True

    checker = _Checker()
    backend = ManiSkillBackend.__new__(ManiSkillBackend)
    backend._checker = checker
    assert backend.success() is True
    assert checker.calls == 1


def test_maniskill_backend_close_swallows_errors():
    from sim.maniskill_backend import ManiSkillBackend

    class _Env:
        def close(self):
            raise RuntimeError("boom")

    backend = ManiSkillBackend.__new__(ManiSkillBackend)
    backend._env = _Env()
    backend.close()
