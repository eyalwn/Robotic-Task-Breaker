"""LIBERO VLA smoke helpers and runnable smoke entrypoints."""
from __future__ import annotations

import inspect
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

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

# LIBERO-10 task 0 — all smoke entrypoints target this scene explicitly.
_SMOKE_TASK_SUITE = "libero_10"
_SMOKE_TASK_IDX = 0


def apply_smoke_env_defaults() -> None:
    os.environ.pop("VLA_AXIS_SIGNS", None)
    os.environ.pop("VLA_AXIS_REMAP", None)
    os.environ.setdefault("VLA_USE_ENV_TASK_SUCCESS", "false")
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
    os.environ.setdefault("VLA_GRIPPER_CONFIDENCE", "0.5")
    os.environ.setdefault("VLA_FAIL_DIST_CONSECUTIVE_STEPS", "200")
    os.environ.setdefault("VLA_FAIL_IK_SATURATION_STEPS", "200")


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


def _bootstrap_smoke() -> None:
    load_dotenv(find_dotenv())
    apply_smoke_env_defaults()
    configure_smoke_logging()


def _close_env(env: Any) -> None:
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception as exc:
            logging.debug("env.close() failed: %s", exc)


def _horizontal_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _object_near_basket(
    env: Any,
    object_body: str,
    basket_body: str,
    max_xy: float,
) -> tuple[bool, float | None]:
    from sim.libero_skills import libero_body_position

    obj = libero_body_position(env, object_body)
    basket = libero_body_position(env, basket_body)
    if obj is None or basket is None:
        return False, None
    xy = _horizontal_dist(obj, basket)
    return xy <= max_xy, xy


def _run_vla(vla_tool: Any, prompt: str, target: str, arm: Any) -> dict | None:
    if not arm.set_target_object(target):
        logging.error("Could not set target %r", target)
        return None
    logging.info("VLA phase: prompt=%r target=%r", prompt, target)
    payload = parse_payload(vla_tool.invoke({"task": prompt}))
    if payload is None:
        return None
    log_payload(prompt, payload)
    return payload


def run_grasp_smoke() -> int:
    """Isolated VLA grasp smoke for LIBERO-10 task 0."""
    from sim.libero_backend import LiberoArm, LiberoCamera, create_env
    from sim.libero_skills import libero_object_human_name
    from vla import VLAConfig, create_vla_manipulation

    _bootstrap_smoke()

    target_body = os.environ.get("LIBERO_SMOKE_TARGET", "alphabet_soup_1_main").strip()
    default_prompt = f"pick up the {libero_object_human_name(target_body)}"
    vla_prompt = (
        os.environ.get("LIBERO_SMOKE_PROMPT", default_prompt).strip()
        or default_prompt
    )
    vla_port = int(os.environ.get("VLA_PORT", "8765"))

    env = None
    try:
        env = create_env(task_suite=_SMOKE_TASK_SUITE, task_idx=_SMOKE_TASK_IDX)
        arm = LiberoArm(env)
        if not arm.set_target_object(target_body):
            logging.error("Could not set target object %r", target_body)
            return 1

        camera = LiberoCamera(env, arm=arm)
        vla_cfg = VLAConfig.from_env()
        vla_tool = create_vla_manipulation(
            tool_name="arm_vla_smoke",
            tool_description="VLA grasp smoke test",
            task_prompt=vla_prompt,
            server_address=f"http://localhost:{vla_port}",
            arm=arm,
            camera=camera,
            cfg=vla_cfg,
        )

        logging.info(
            "Running VLA grasp smoke: prompt=%r target=%r", vla_prompt, target_body
        )
        payload = parse_payload(vla_tool.invoke({"task": vla_prompt}))
        if payload is None:
            return 1

        min_dist = payload.get("min_dist")
        steps = payload.get("steps_applied")
        reason = payload.get("reason")
        grasp_confirmed = payload.get("grasp_confirmed")
        logging.info(
            "Smoke result: reason=%s steps=%s min_dist=%s grasp_confirmed=%s payload=%s",
            reason,
            steps,
            min_dist,
            grasp_confirmed,
            json.dumps(payload, separators=(",", ":")),
        )

        if grasp_passed(payload):
            logging.info(
                "PASS: grasp confirmed (reason=%s grasp_confirmed=%s min_dist=%s)",
                reason,
                grasp_confirmed,
                min_dist,
            )
            return 0

        logging.error(
            "FAIL: no confirmed lift (reason=%s grasp_confirmed=%s min_dist=%s)",
            reason,
            grasp_confirmed,
            min_dist,
        )
        return 1
    finally:
        _close_env(env)


def run_place_smoke() -> int:
    """Isolated VLA place smoke for LIBERO-10 task 0."""
    from sim.libero_backend import LiberoArm, LiberoCamera, create_env
    from sim.libero_skills import libero_object_human_name
    from vla import VLAConfig, create_vla_manipulation

    _bootstrap_smoke()

    object_body = os.environ.get("LIBERO_SMOKE_TARGET", "alphabet_soup_1_main").strip()
    basket_body = "basket_1_main"
    human_name = libero_object_human_name(object_body)
    grab_prompt = f"pick up the {human_name}"
    place_prompt = f"put the {human_name} in the basket"
    vla_port = int(os.environ.get("VLA_PORT", "8765"))
    skip_grab = os.environ.get("LIBERO_SMOKE_SKIP_GRAB", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    max_xy = float(os.environ.get("LIBERO_PLACE_SMOKE_MAX_XY", "0.12"))

    env = None
    try:
        env = create_env(task_suite=_SMOKE_TASK_SUITE, task_idx=_SMOKE_TASK_IDX)
        arm = LiberoArm(env)
        camera = LiberoCamera(env, arm=arm)
        vla_cfg = VLAConfig.from_env()
        vla_tool = create_vla_manipulation(
            tool_name="arm_vla_place_smoke",
            tool_description="VLA place smoke test",
            task_prompt=place_prompt,
            server_address=f"http://localhost:{vla_port}",
            arm=arm,
            camera=camera,
            cfg=vla_cfg,
        )

        if not skip_grab:
            if not arm.set_target_object(object_body):
                logging.error("Could not set grab target %r", object_body)
                return 1
            logging.info(
                "Running grab phase: prompt=%r target=%r", grab_prompt, object_body
            )
            grab_payload = parse_payload(vla_tool.invoke({"task": grab_prompt}))
            if grab_payload is None:
                return 1
            log_payload("Grab result", grab_payload)
            if not grasp_passed(grab_payload):
                logging.error(
                    "FAIL: grab did not confirm lift (reason=%s grasp_confirmed=%s)",
                    grab_payload.get("reason"),
                    grab_payload.get("grasp_confirmed"),
                )
                return 1
            logging.info("Grab PASS — proceeding to place")
        else:
            logging.info("Skipping grab (LIBERO_SMOKE_SKIP_GRAB=1)")

        if not arm.set_target_object(basket_body):
            logging.error("Could not set place target %r", basket_body)
            return 1

        logging.info(
            "Running place phase: prompt=%r target=%r", place_prompt, basket_body
        )
        place_payload = parse_payload(vla_tool.invoke({"task": place_prompt}))
        if place_payload is None:
            return 1
        log_payload("Place result", place_payload)

        near, xy_dist = _object_near_basket(env, object_body, basket_body, max_xy)
        logging.info(
            "Place geometry: object=%r basket=%r xy_dist=%s near=%s",
            object_body,
            basket_body,
            f"{xy_dist:.4f}" if xy_dist is not None else None,
            near,
        )

        if near:
            logging.info(
                "PASS: %s near basket (xy_dist=%s min_dist=%s steps=%s)",
                object_body,
                f"{xy_dist:.4f}" if xy_dist is not None else None,
                place_payload.get("min_dist"),
                place_payload.get("steps_applied"),
            )
            return 0

        logging.error(
            "FAIL: object not in basket region (xy_dist=%s min_dist=%s steps=%s reason=%s)",
            f"{xy_dist:.4f}" if xy_dist is not None else None,
            place_payload.get("min_dist"),
            place_payload.get("steps_applied"),
            place_payload.get("reason"),
        )
        return 1
    finally:
        _close_env(env)


def run_sequential_smoke() -> int:
    """Sequential VLA smoke: soup grab→place→prepare→cream grab."""
    from sim.libero_backend import LiberoArm, LiberoCamera, create_env
    from sim.libero_skills import (
        libero_body_position,
        libero_horizontal_xy_dist,
        libero_object_human_name,
    )
    from vla import VLAConfig, create_vla_manipulation

    _bootstrap_smoke()

    soup_body = "alphabet_soup_1_main"
    cream_body = "cream_cheese_1_main"
    basket_body = "basket_1_main"
    soup_human = libero_object_human_name(soup_body)
    cream_human = libero_object_human_name(cream_body)
    soup_grab = f"pick up the {soup_human}"
    soup_place = f"put the {soup_human} in the basket"
    cream_grab = f"pick up the {cream_human}"
    vla_port = int(os.environ.get("VLA_PORT", "8765"))

    env = None
    try:
        env = create_env(task_suite=_SMOKE_TASK_SUITE, task_idx=_SMOKE_TASK_IDX)
        arm = LiberoArm(env)
        camera = LiberoCamera(env, arm=arm)
        vla_cfg = VLAConfig.from_env()
        vla_tool = create_vla_manipulation(
            tool_name="arm_vla_sequential_smoke",
            tool_description="VLA sequential grab-place-grab smoke",
            task_prompt=soup_place,
            server_address=f"http://localhost:{vla_port}",
            arm=arm,
            camera=camera,
            cfg=vla_cfg,
        )

        soup_grab_payload = _run_vla(vla_tool, soup_grab, soup_body, arm)
        if soup_grab_payload is None or not grasp_passed(soup_grab_payload):
            logging.error("FAIL: soup grab did not confirm lift")
            return 1

        soup_place_payload = _run_vla(vla_tool, soup_place, basket_body, arm)
        if soup_place_payload is None:
            return 1

        if not arm.set_target_object(cream_body):
            logging.error("Could not set target %r before prepare", cream_body)
            return 1

        log_arm_state(arm, "before prepare_for_grab")
        logging.info("Running prepare_for_grab between place and second grab")
        arm.prepare_for_grab(next_object=cream_body)
        log_arm_state(arm, "after prepare_for_grab")
        cream_pos = libero_body_position(env, cream_body)
        eef = arm.last_tcp_pos
        if cream_pos and eef:
            horiz = libero_horizontal_xy_dist(eef, cream_pos)
            logging.info(
                "cream VLA start: eef_xyz=%s object_xyz=%s horiz_dist=%.3f",
                eef,
                cream_pos,
                horiz,
            )
        log_arm_state(arm, "before cream VLA")

        cream_grab_payload = _run_vla(vla_tool, cream_grab, cream_body, arm)
        if cream_grab_payload is None:
            return 1

        if grasp_passed(cream_grab_payload):
            logging.info(
                "PASS: cream cheese grasp confirmed (reason=%s grasp_confirmed=%s min_dist=%s)",
                cream_grab_payload.get("reason"),
                cream_grab_payload.get("grasp_confirmed"),
                cream_grab_payload.get("min_dist"),
            )
            return 0

        logging.error(
            "FAIL: cream grab not confirmed (reason=%s grasp_confirmed=%s min_dist=%s)",
            cream_grab_payload.get("reason"),
            cream_grab_payload.get("grasp_confirmed"),
            cream_grab_payload.get("min_dist"),
        )
        return 1
    finally:
        _close_env(env)
