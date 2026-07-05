"""Shared helpers for LIBERO VLA smoke scripts (examples/)."""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_RUN_LOG_LOGGER_NAMES = ("vla.tools", "sim.libero_backend", "core.plan")
_NOISY_LOGGERS = (
    "numba",
    "numexpr",
    "OpenGL",
    "OpenGL.acceleratesupport",
    "robosuite",
    "datasets",
    "jax",
    "jaxlib",
)
_run_log_attached_path: str | None = None

_LIFT_REASONS = frozenset({"grasp_lift", "proximity_grasp", "task_success"})


def apply_smoke_env_defaults() -> None:
    os.environ.pop("VLA_AXIS_SIGNS", None)
    os.environ.pop("VLA_AXIS_REMAP", None)
    os.environ.setdefault("VLA_USE_ENV_TASK_SUCCESS", "false")
    os.environ.setdefault("LIBERO_TASK_SUITE", "libero_10")
    os.environ.setdefault("LIBERO_TASK_IDX", "0")
    os.environ.setdefault("VLA_GRIPPER_MODE", "absolute_open_close")
    os.environ.setdefault("VLA_ACTION_SCALE", "1.0")
    os.environ.setdefault("LIBERO_EE_DELTA_SCALE", "1.0")
    os.environ.setdefault("LIBERO_VLA_IMAGE_FLIP", "1")
    os.environ.setdefault("VLA_STEPS_PER_ACTION", "1")
    os.environ.setdefault("VLA_ADAPTIVE_SCALE", "false")
    os.environ.setdefault("VLA_CHUNK_STEPS", "5")
    os.environ.setdefault("VLA_STAGNATION_CHUNKS", "20")
    os.environ.setdefault("LIBERO_SETTLE_STEPS", "10")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_DESCEND_DZ", "0.35")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_DESCEND_STEPS", "20")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_XY_RETREAT_DELTA", "0.15")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_XY_RETREAT_STEPS", "15")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_TARGET_Z", "0.55")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_LOW_Z_MAX_STEPS", "30")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_APPROACH_MAX_STEPS", "80")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_APPROACH_DELTA", "0.15")
    os.environ.setdefault("LIBERO_PREPARE_GRAB_APPROACH_RADIUS", "0.15")
    os.environ.setdefault("VLA_MAX_INNER_STEPS", "200")
    os.environ.setdefault("VLA_MAX_TOOL_SECONDS", "180")
    os.environ.setdefault("VLA_SIM_PICKCUBE_GRIPPER_PROFILE", "false")
    os.environ.setdefault("VLA_FAIL_WORSENING_STEPS", "200")
    os.environ.setdefault("VLA_PROXIMITY_GRASP", "false")


def attach_run_log_file(
    extra_logger_names: tuple[str, ...] = (),
) -> None:
    """When LONGOR_RUN_LOG is set, mirror target loggers at DEBUG to that path."""
    global _run_log_attached_path
    path = os.environ.get("LONGOR_RUN_LOG", "").strip()
    if not path:
        return
    if _run_log_attached_path == path:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(levelname)s [%(name)s] %(message)s")
    )
    names = _RUN_LOG_LOGGER_NAMES + extra_logger_names
    for name in names:
        logger = logging.getLogger(name)
        if handler not in logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root = logging.getLogger()
    resolved = str(log_path.resolve())
    for root_handler in list(root.handlers):
        if (
            isinstance(root_handler, logging.FileHandler)
            and getattr(root_handler, "baseFilename", None) == resolved
        ):
            root.removeHandler(root_handler)
    _run_log_attached_path = path


def configure_smoke_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for _h in logging.root.handlers:
        if isinstance(_h, logging.StreamHandler) and not isinstance(
            _h, logging.FileHandler
        ):
            _h.setLevel(logging.INFO)
    for name in ("openai", "httpcore", "httpx", "trimesh", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    caller_name = "__main__"
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        caller_name = frame.f_back.f_globals.get("__name__", "__main__")
    attach_run_log_file(extra_logger_names=(caller_name,))


def parse_payload(result: str) -> dict | None:
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        logging.error("VLA tool returned non-JSON: %s", result)
        return None


def grasp_passed(payload: dict) -> bool:
    if payload.get("grasp_confirmed") is True:
        return True
    return (
        payload.get("reason") in _LIFT_REASONS
        and payload.get("status") == "success"
    )


def log_payload(label: str, payload: dict) -> None:
    logging.info(
        "%s: reason=%s steps=%s min_dist=%s status=%s payload=%s",
        label,
        payload.get("reason"),
        payload.get("steps_applied"),
        payload.get("min_dist"),
        payload.get("status"),
        json.dumps(payload, separators=(",", ":")),
    )


def log_arm_state(arm: Any, label: str) -> None:
    pos = getattr(arm, "last_tcp_pos", None)
    grip = getattr(arm, "_gripper_value", None)
    logging.info("%s: eef_xyz=%s gripper=%s", label, pos, grip)
