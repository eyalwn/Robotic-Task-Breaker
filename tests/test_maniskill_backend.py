"""Tests for sim.maniskill_backend lazy-import behaviour."""
import types
import sys

import pytest


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
			maniskill_backend.create_env()
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
