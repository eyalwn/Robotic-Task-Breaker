"""Assertion tests for sim camera placeholder identity and degraded-vision logging."""
from __future__ import annotations

import logging

from conftest import FakeEnv
from sim.camera_base import _make_placeholder_jpeg
from sim.libero_backend import LiberoCamera
from sim.maniskill_backend import ManiSkillCamera


def test_placeholder_jpeg_identity_cross_backend_exclusivity():
    assert ManiSkillCamera._PLACEHOLDER_TEXT != LiberoCamera._PLACEHOLDER_TEXT

    cam_ms = ManiSkillCamera(FakeEnv())
    cam_lb = LiberoCamera(FakeEnv())

    ms_ph = _make_placeholder_jpeg(ManiSkillCamera._PLACEHOLDER_TEXT)
    lb_ph = _make_placeholder_jpeg(LiberoCamera._PLACEHOLDER_TEXT)

    assert cam_ms.is_placeholder_jpeg(ms_ph) is True
    assert cam_ms.is_placeholder_jpeg(lb_ph) is False
    assert cam_lb.is_placeholder_jpeg(lb_ph) is True
    assert cam_lb.is_placeholder_jpeg(ms_ph) is False


def test_maniskill_mark_vision_degraded_logs_error_once(caplog):
    caplog.set_level(logging.ERROR)
    for name in ("sim.maniskill_backend", "sim.camera_base"):
        logging.getLogger(name).propagate = True

    cam = ManiSkillCamera(FakeEnv())
    cam._mark_vision_degraded("sentinel_reason")

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) >= 1
    assert "sentinel_reason" in caplog.text

    n_after_first = len(errors)
    cam._mark_vision_degraded("second_call")
    errors_after = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors_after) == n_after_first
    assert "second_call" not in caplog.text


def test_libero_mark_vision_degraded_logs_error_once(caplog):
    caplog.set_level(logging.ERROR)
    for name in ("sim.libero_backend", "sim.camera_base"):
        logging.getLogger(name).propagate = True

    cam = LiberoCamera(FakeEnv())
    cam._mark_vision_degraded("sentinel_reason")

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) >= 1
    assert "sentinel_reason" in caplog.text

    n_after_first = len(errors)
    cam._mark_vision_degraded("second_call")
    errors_after = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors_after) == n_after_first
    assert "second_call" not in caplog.text


def test_maniskill_log_vision_degraded_message_includes_vulkan_keyword(caplog):
    """Backend override mentions Vulkan / DISPLAY_ENV style remediation (stable substring)."""
    caplog.set_level(logging.ERROR)
    logging.getLogger("sim.maniskill_backend").propagate = True

    cam = ManiSkillCamera(FakeEnv())
    cam._mark_vision_degraded("r")
    assert "Vulkan" in caplog.text


def test_libero_log_vision_degraded_message_includes_libero_keyword(caplog):
    caplog.set_level(logging.ERROR)
    logging.getLogger("sim.libero_backend").propagate = True

    cam = LiberoCamera(FakeEnv())
    cam._mark_vision_degraded("r")
    assert "LIBERO" in caplog.text
