"""Sequential VLA smoke: soup grab→place→prepare→cream grab (no LLM).

Reproduces §10 second-grab failure after place when arm stays high over basket.
PASS = cream cheese grasp_confirmed=true (same gate as §9).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_examples = Path(__file__).resolve().parent
_src = _examples.parent / "src"
for path in (_src, _examples):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

load_dotenv(find_dotenv())

from libero_vla_smoke_common import (
    apply_smoke_env_defaults,
    configure_smoke_logging,
    grasp_passed,
    log_arm_state,
    log_payload,
    parse_payload,
)

apply_smoke_env_defaults()
configure_smoke_logging()

from sim.libero_backend import LiberoArm, LiberoCamera, create_env
from sim.libero_skills import (
    libero_body_position,
    libero_horizontal_xy_dist,
    libero_object_human_name,
)
from vla import VLAConfig, create_vla_manipulation

SOUP_BODY = "alphabet_soup_1_main"
CREAM_BODY = "cream_cheese_1_main"
BASKET_BODY = "basket_1_main"
SOUP_HUMAN = libero_object_human_name(SOUP_BODY)
CREAM_HUMAN = libero_object_human_name(CREAM_BODY)
SOUP_GRAB = f"pick up the {SOUP_HUMAN}"
SOUP_PLACE = f"put the {SOUP_HUMAN} in the basket"
CREAM_GRAB = f"pick up the {CREAM_HUMAN}"
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))


def _run_vla(vla_tool, prompt: str, target: str, arm: LiberoArm) -> dict | None:
    if not arm.set_target_object(target):
        logging.error("Could not set target %r", target)
        return None
    logging.info("VLA phase: prompt=%r target=%r", prompt, target)
    payload = parse_payload(vla_tool.invoke({"task": prompt}))
    if payload is None:
        return None
    log_payload(prompt, payload)
    return payload


def main() -> int:
    env = None
    try:
        env = create_env()
        arm = LiberoArm(env)
        camera = LiberoCamera(env, arm=arm)
        vla_cfg = VLAConfig.from_env()
        vla_tool = create_vla_manipulation(
            tool_name="arm_vla_sequential_smoke",
            tool_description="VLA sequential grab-place-grab smoke",
            task_prompt=SOUP_PLACE,
            server_address=f"http://localhost:{VLA_PORT}",
            arm=arm,
            camera=camera,
            cfg=vla_cfg,
        )

        soup_grab = _run_vla(vla_tool, SOUP_GRAB, SOUP_BODY, arm)
        if soup_grab is None or not grasp_passed(soup_grab):
            logging.error("FAIL: soup grab did not confirm lift")
            return 1

        soup_place = _run_vla(vla_tool, SOUP_PLACE, BASKET_BODY, arm)
        if soup_place is None:
            return 1

        if not arm.set_target_object(CREAM_BODY):
            logging.error("Could not set target %r before prepare", CREAM_BODY)
            return 1

        log_arm_state(arm, "before prepare_for_grab")
        logging.info("Running prepare_for_grab between place and second grab")
        arm.prepare_for_grab(next_object=CREAM_BODY)
        log_arm_state(arm, "after prepare_for_grab")
        cream_pos = libero_body_position(env, CREAM_BODY)
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

        cream_grab = _run_vla(vla_tool, CREAM_GRAB, CREAM_BODY, arm)
        if cream_grab is None:
            return 1

        if grasp_passed(cream_grab):
            logging.info(
                "PASS: cream cheese grasp confirmed (reason=%s grasp_confirmed=%s min_dist=%s)",
                cream_grab.get("reason"),
                cream_grab.get("grasp_confirmed"),
                cream_grab.get("min_dist"),
            )
            return 0

        logging.error(
            "FAIL: cream grab not confirmed (reason=%s grasp_confirmed=%s min_dist=%s)",
            cream_grab.get("reason"),
            cream_grab.get("grasp_confirmed"),
            cream_grab.get("min_dist"),
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
