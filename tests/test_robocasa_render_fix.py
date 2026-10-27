"""Hermetic tests for the robosuite render-context finalisation fix."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from envs import robocasa_env


def _module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _register(monkeypatch, *mods: ModuleType) -> None:
    for mod in mods:
        monkeypatch.setitem(sys.modules, mod.__name__, mod)


class _FakeEgl:
    """One current context per thread; ``free_in`` records who was current."""

    EGL_NO_SURFACE = None

    def __init__(self) -> None:
        self.current = None
        self.free_in: list = []

    def eglGetCurrentContext(self):
        return self.current

    def eglMakeCurrent(self, display, draw, read, ctx) -> bool:
        self.current = ctx
        return True


def _install(monkeypatch):
    """Fake EGL plus a stock-like ``MjRenderContext`` whose ``__del__`` is naive."""
    egl = _FakeEgl()

    class MjRenderContext:
        def __init__(self, address: int) -> None:
            handle = SimpleNamespace(address=address)

            def make() -> None:  # no reference to self: freed on ``del``
                egl.eglMakeCurrent(None, None, None, handle)

            self.gl_ctx = SimpleNamespace(_context=handle, make_current=make)

        def __del__(self) -> None:
            cur = egl.current
            egl.free_in.append(cur.address if cur else None)
            egl.current = None  # gl_ctx.free() ends with eglReleaseThread()

    _register(
        monkeypatch, _module("mujoco"), _module("mujoco.egl", egl_ext=egl),
        _module("robosuite"), _module("robosuite.renderers"),
        _module("robosuite.renderers.context",
                egl_context=SimpleNamespace(EGL_DISPLAY="display")),
        _module("robosuite.utils"),
        _module("robosuite.utils.binding_utils", MjRenderContext=MjRenderContext))
    return egl, MjRenderContext


def test_stale_context_is_freed_in_its_own_context(monkeypatch) -> None:
    egl, ctx_cls = _install(monkeypatch)
    robocasa_env._free_render_contexts_in_own_egl_context()
    live, stale = ctx_cls(1), ctx_cls(2)
    egl.current = live.gl_ctx._context
    del stale
    assert egl.free_in == [2]
    assert egl.current is live.gl_ctx._context


def test_current_context_is_freed_without_rebinding(monkeypatch) -> None:
    egl, ctx_cls = _install(monkeypatch)
    robocasa_env._free_render_contexts_in_own_egl_context()
    ctx = ctx_cls(1)
    egl.current = ctx.gl_ctx._context
    del ctx
    assert egl.free_in == [1]
    assert egl.current is None


def test_fix_is_applied_once(monkeypatch) -> None:
    _egl, ctx_cls = _install(monkeypatch)
    robocasa_env._free_render_contexts_in_own_egl_context()
    patched = ctx_cls.__del__
    robocasa_env._free_render_contexts_in_own_egl_context()
    assert ctx_cls.__del__ is patched


def test_terminated_egl_falls_back_to_plain_finalisation(monkeypatch) -> None:
    egl, ctx_cls = _install(monkeypatch)
    robocasa_env._free_render_contexts_in_own_egl_context()
    live, stale = ctx_cls(1), ctx_cls(2)

    def refuse() -> None:
        raise RuntimeError("Failed to make the EGL context current.")

    stale.gl_ctx.make_current = refuse
    egl.current = live.gl_ctx._context
    del stale
    assert egl.free_in == [1]  # freed wherever it was current, as stock does
    assert egl.current is None


def test_import_robocasa_gym_applies_the_fix(monkeypatch) -> None:
    gym = _module("gymnasium")
    _register(monkeypatch, gym, _module("robosuite"), _module("robocasa"),
              _module("robocasa.utils"),
              _module("robocasa.utils.gym_utils", GrootRoboCasaEnv=object))
    calls: list[int] = []
    monkeypatch.setattr(robocasa_env, "_free_render_contexts_in_own_egl_context",
                        lambda: calls.append(1))
    assert robocasa_env._import_robocasa_gym() is gym
    assert calls == [1]
