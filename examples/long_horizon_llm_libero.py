"""M6 - LLM-emitted plan on LIBERO-Long (LIBERO-10) with full executor + VLA."""
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
os.environ.pop("VLA_AXIS_SIGNS", None)
os.environ.pop("VLA_AXIS_REMAP", None)
os.environ.setdefault("VLA_USE_ENV_TASK_SUCCESS", "false")
os.environ.setdefault("VLA_GRIPPER_MODE", "absolute_open_close")
os.environ.setdefault("VLA_ACTION_SCALE", "1.0")
os.environ.setdefault("LIBERO_EE_DELTA_SCALE", "1.0")
os.environ.setdefault("VLA_STEPS_PER_ACTION", "1")
os.environ.setdefault("VLA_ADAPTIVE_SCALE", "false")
os.environ.setdefault("VLA_CHUNK_STEPS", "5")
os.environ.setdefault("VLA_STAGNATION_CHUNKS", "20")
os.environ.setdefault("LIBERO_SETTLE_STEPS", "10")
os.environ.setdefault("VLA_FAIL_WORSENING_STEPS", "200")
os.environ.setdefault("VLA_PROXIMITY_GRASP", "false")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
    force=True,
)
for _h in logging.root.handlers:
    if isinstance(_h, logging.StreamHandler) and not isinstance(
        _h, logging.FileHandler
    ):
        _h.setLevel(logging.INFO)
for _noisy in ("openai", "httpcore", "httpx", "trimesh", "matplotlib", "numexpr"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from libero_vla_smoke_common import attach_run_log_file

attach_run_log_file(extra_logger_names=(__name__,))

from core.plan import Executor, LLMFeedbackEvaluator, LLMPlanner
from core.skills import SkillRegistry
from sim.libero_backend import LiberoArm, LiberoCamera, create_env
from sim.libero_skills import (
    libero_body_position,
    libero_horizontal_xy_dist,
    register_libero_skills,
)
from sim.libero_success_checker import plan_return_indicates_failure
from vla import VLAConfig, create_vla_manipulation
from vla.arm_skills import register_arm_skills

MODEL = os.environ.get("LLM_MODEL")
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
TASK_SUITE = os.environ.get("LIBERO_TASK_SUITE", "libero_10")
TASK_IDX = int(os.environ.get("LIBERO_TASK_IDX", "0"))
TRACE_DIR = os.environ.get("LONGOR_TRACE_DIR", "traces")
MAX_REPLANS = int(os.environ.get("LONGOR_MAX_REPLANS", "3"))
STEP_BUDGET = int(os.environ.get("LONGOR_STEP_BUDGET", "30"))

# LIBERO-10 task 0 canonical language instruction.
GOAL = "put the alphabet soup can and the cream cheese box in the basket"

_TASK_OBJECTS = ("alphabet_soup_1_main", "cream_cheese_1_main")


def _log_final_scene(env, arm) -> bool:
    step = getattr(arm, "step", None)
    if callable(step):
        step()
    info = getattr(arm, "last_info", {}) or {}
    raw = info.get("is_success")
    if raw is None:
        raw = info.get("success")
    sim_success = False
    if raw is not None:
        try:
            import numpy as np

            if isinstance(raw, np.ndarray):
                sim_success = bool(raw.flat[0])
            else:
                sim_success = bool(raw)
        except Exception:
            sim_success = False
    basket_pos = libero_body_position(env, "basket_1_main")
    xy_dists: dict[str, float | None] = {}
    for name in _TASK_OBJECTS:
        pos = libero_body_position(env, name)
        xy = pos[:2] if pos is not None else None
        logging.info("Final object %s xy=%s", name, xy)
        if pos is not None and basket_pos is not None:
            xy_dists[name] = libero_horizontal_xy_dist(pos, basket_pos)
        else:
            xy_dists[name] = None
    logging.debug(
        "LIBERO task result: is_success=%s geometry_xy_dist=%s basket_pos=%s",
        sim_success,
        xy_dists,
        basket_pos,
    )
    return sim_success


def main() -> None:
    logging.info("LLM model: %s", MODEL)
    logging.info(
        "VLA: %s @ http://localhost:%s",
        os.environ.get("VLA_POLICY", "?"),
        VLA_PORT,
    )
    env = create_env(task_suite=TASK_SUITE, task_idx=TASK_IDX)
    arm = LiberoArm(env)
    camera = LiberoCamera(env, arm=arm)

    vla_cfg = VLAConfig.from_env()
    vla_lc_tool = create_vla_manipulation(
        tool_name="arm_vla",
        tool_description="VLA-driven arm manipulation",
        task_prompt=GOAL,
        server_address=f"http://localhost:{VLA_PORT}",
        arm=arm,
        camera=camera,
        max_steps=vla_cfg.max_inner_steps,
        cfg=vla_cfg,
    )
    vla_callable = lambda task: vla_lc_tool.invoke({"task": task})

    registry = SkillRegistry()
    register_libero_skills(registry, env)
    register_arm_skills(registry, vla_callable, arm=arm)

    objects = registry.get("state.list_objects")()
    logging.info("Scene objects: %s", objects)

    enriched_goal = GOAL
    if objects:
        enriched_goal += (
            "\n\nAvailable scene objects "
            "(use these exact names in all skill args): "
            + ", ".join(str(o) for o in objects)
        )

    planner = LLMPlanner(model=MODEL, max_plan_retries=3)
    feedback_eval = LLMFeedbackEvaluator(model=MODEL)

    executor = Executor(
        registry=registry,
        planner=planner,
        feedback_evaluator=feedback_eval,
        image_fn=camera.capture_image,
        max_replans=MAX_REPLANS,
        step_budget=STEP_BUDGET,
        trace_dir=TRACE_DIR,
    )

    catalog = registry.render_catalog()
    initial_plan = planner.plan(enriched_goal, catalog)
    logging.info("Initial plan received.")

    try:
        result = executor.run(goal=enriched_goal, plan=initial_plan)
    finally:
        sim_success = _log_final_scene(env, arm)
        env.close()

    if result.status == "done":
        logging.info("Task COMPLETED: %r", enriched_goal.split("\n")[0])
    else:
        logging.warning(
            "Task FAILED (status=%s): %r",
            result.status,
            enriched_goal.split("\n")[0],
        )

    exit_code = 0
    if result.status == "replan_failed":
        logging.error(
            "Run stopped: replan failed after %d replan(s), %d steps — %s",
            result.replans,
            result.steps,
            result.error,
        )
        exit_code = 1
    elif result.status != "done":
        logging.warning(
            "Run finished with status=%s replans=%d steps=%d error=%r",
            result.status,
            result.replans,
            result.steps,
            result.error,
        )
        if sim_success:
            logging.error(
                "Executor status=%s but LIBERO sim success is true — stale or partial success",
                result.status,
            )
        exit_code = 1
    else:
        if result.value is not None and isinstance(result.value, str):
            logging.info(
                "Run finished with return value=%r (status=%s replans=%d steps=%d)",
                result.value,
                result.status,
                result.replans,
                result.steps,
            )
        else:
            logging.info(
                "Run complete: status=%s replans=%d steps=%d value=%r",
                result.status,
                result.replans,
                result.steps,
                result.value,
            )
        if plan_return_indicates_failure(result.value):
            logging.error(
                "Executor status=done but return value indicates failure: %r",
                result.value,
            )
            exit_code = 1
        elif not sim_success:
            logging.error(
                "Executor status=done but LIBERO sim success is false — task not achieved"
            )
            exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
