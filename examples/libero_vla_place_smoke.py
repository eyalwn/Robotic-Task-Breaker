"""Isolated VLA place smoke for LIBERO-10 task 0 (no LLM).

Runs grab then place with LIBERO-native prompt:
  put the alphabet soup can in the basket

Gate: object near basket region (xy) after place, or logs min_dist/steps on failure.

Env overrides:
  LIBERO_SMOKE_TARGET — object body (default alphabet_soup_1_main)
  LIBERO_SMOKE_SKIP_GRAB — if 1, skip grab (assume object already held)
  LIBERO_PLACE_SMOKE_MAX_XY — max horizontal object→basket distance for PASS (default 0.12)
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_examples = Path(__file__).resolve().parent
if str(_examples) not in sys.path:
    sys.path.insert(0, str(_examples))

load_dotenv(find_dotenv())

from libero_vla_smoke_common import (
    apply_smoke_env_defaults,
    configure_smoke_logging,
    grasp_passed,
    log_payload,
    parse_payload,
)

apply_smoke_env_defaults()
configure_smoke_logging()

from sim.libero_backend import LiberoArm, LiberoCamera, create_env
from sim.libero_skills import libero_body_position, libero_object_human_name
from vla import VLAConfig, create_vla_manipulation

OBJECT_BODY = os.environ.get("LIBERO_SMOKE_TARGET", "alphabet_soup_1_main").strip()
BASKET_BODY = "basket_1_main"
HUMAN_NAME = libero_object_human_name(OBJECT_BODY)
GRAB_PROMPT = f"pick up the {HUMAN_NAME}"
PLACE_PROMPT = f"put the {HUMAN_NAME} in the basket"
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
SKIP_GRAB = os.environ.get("LIBERO_SMOKE_SKIP_GRAB", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
MAX_XY = float(os.environ.get("LIBERO_PLACE_SMOKE_MAX_XY", "0.12"))


def _horizontal_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _object_near_basket(env, object_body: str, basket_body: str) -> tuple[bool, float | None]:
    obj = libero_body_position(env, object_body)
    basket = libero_body_position(env, basket_body)
    if obj is None or basket is None:
        return False, None
    xy = _horizontal_dist(obj, basket)
    return xy <= MAX_XY, xy


def main() -> int:
    env = None
    try:
        env = create_env()
        arm = LiberoArm(env)
        camera = LiberoCamera(env, arm=arm)
        vla_cfg = VLAConfig.from_env()
        vla_tool = create_vla_manipulation(
            tool_name="arm_vla_place_smoke",
            tool_description="VLA place smoke test",
            task_prompt=PLACE_PROMPT,
            server_address=f"http://localhost:{VLA_PORT}",
            arm=arm,
            camera=camera,
            cfg=vla_cfg,
        )

        if not SKIP_GRAB:
            if not arm.set_target_object(OBJECT_BODY):
                logging.error("Could not set grab target %r", OBJECT_BODY)
                return 1
            logging.info("Running grab phase: prompt=%r target=%r", GRAB_PROMPT, OBJECT_BODY)
            grab_payload = parse_payload(vla_tool.invoke({"task": GRAB_PROMPT}))
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

        if not arm.set_target_object(BASKET_BODY):
            logging.error("Could not set place target %r", BASKET_BODY)
            return 1

        logging.info(
            "Running place phase: prompt=%r target=%r",
            PLACE_PROMPT,
            BASKET_BODY,
        )
        place_payload = parse_payload(vla_tool.invoke({"task": PLACE_PROMPT}))
        if place_payload is None:
            return 1
        log_payload("Place result", place_payload)

        near, xy_dist = _object_near_basket(env, OBJECT_BODY, BASKET_BODY)
        logging.info(
            "Place geometry: object=%r basket=%r xy_dist=%s near=%s",
            OBJECT_BODY,
            BASKET_BODY,
            f"{xy_dist:.4f}" if xy_dist is not None else None,
            near,
        )

        if near:
            logging.info(
                "PASS: %s near basket (xy_dist=%s min_dist=%s steps=%s)",
                OBJECT_BODY,
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
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception as exc:
                logging.debug("env.close() failed: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
