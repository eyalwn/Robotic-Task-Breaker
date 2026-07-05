"""Isolated VLA grasp smoke test for LIBERO-10 task 0 (no LLM).

Gate: confirmed object lift (grasp_confirmed=true / reason=grasp_lift).

Env overrides:
  LIBERO_SMOKE_TARGET — MuJoCo body name (default alphabet_soup_1_main)
  LIBERO_SMOKE_PROMPT — VLA task string (default derived from target)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

load_dotenv(find_dotenv())
# PickCube axis calibration must not bleed from .env into LIBERO OpenPI path.
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
os.environ.setdefault("VLA_MAX_INNER_STEPS", "200")
os.environ.setdefault("VLA_MAX_TOOL_SECONDS", "180")
os.environ.setdefault("VLA_SIM_PICKCUBE_GRIPPER_PROFILE", "false")
os.environ.setdefault("VLA_FAIL_WORSENING_STEPS", "200")
os.environ.setdefault("VLA_PROXIMITY_GRASP", "false")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
    force=True,
)
for _noisy in ("openai", "httpcore", "httpx", "trimesh", "matplotlib"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from sim.libero_backend import LiberoArm, LiberoCamera, create_env
from sim.libero_skills import libero_object_human_name
from vla import VLAConfig, create_vla_manipulation

TARGET_BODY = os.environ.get("LIBERO_SMOKE_TARGET", "alphabet_soup_1_main").strip()
_default_prompt = f"pick up the {libero_object_human_name(TARGET_BODY)}"
VLA_PROMPT = os.environ.get("LIBERO_SMOKE_PROMPT", _default_prompt).strip() or _default_prompt
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))

_LIFT_REASONS = frozenset({"grasp_lift", "proximity_grasp", "task_success"})


def _smoke_passed(payload: dict) -> bool:
    if payload.get("grasp_confirmed") is True:
        return True
    reason = payload.get("reason")
    return reason in _LIFT_REASONS and payload.get("status") == "success"


def main() -> int:
    env = None
    try:
        env = create_env()
        arm = LiberoArm(env)
        if not arm.set_target_object(TARGET_BODY):
            logging.error("Could not set target object %r", TARGET_BODY)
            return 1

        camera = LiberoCamera(env, arm=arm)
        vla_cfg = VLAConfig.from_env()
        vla_tool = create_vla_manipulation(
            tool_name="arm_vla_smoke",
            tool_description="VLA grasp smoke test",
            task_prompt=VLA_PROMPT,
            server_address=f"http://localhost:{VLA_PORT}",
            arm=arm,
            camera=camera,
            cfg=vla_cfg,
        )

        logging.info("Running VLA grasp smoke: prompt=%r target=%r", VLA_PROMPT, TARGET_BODY)
        result = vla_tool.invoke({"task": VLA_PROMPT})

        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            logging.error("VLA tool returned non-JSON: %s", result)
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
            result,
        )

        if _smoke_passed(payload):
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
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception as exc:
                logging.debug("env.close() failed: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
