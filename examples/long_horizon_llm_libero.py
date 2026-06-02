"""M6 - LLM-emitted plan on LIBERO-Long (LIBERO-10) with full executor + VLA."""
import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

load_dotenv(find_dotenv())
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
    force=True,
)
for _noisy in ("openai", "httpcore", "httpx", "trimesh", "matplotlib"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from core.plan import Executor, LLMFeedbackEvaluator, LLMPlanner
from core.skills import SkillRegistry
from sim.libero_backend import LiberoArm, LiberoCamera, create_env
from sim.libero_skills import register_libero_skills
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


def main() -> None:
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
        cfg=vla_cfg,
    )
    vla_callable = lambda task: vla_lc_tool.invoke({"task": task})

    registry = SkillRegistry()
    register_libero_skills(registry, env)
    register_arm_skills(registry, vla_callable)

    objects = registry.get("state.list_objects")()
    logging.info("Scene objects: %s", objects)

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
    initial_plan = planner.plan(GOAL, catalog)
    logging.info("Initial plan received.")

    try:
        result = executor.run(goal=GOAL, plan=initial_plan)
    finally:
        env.close()

    logging.info(
        "Run complete: status=%s replans=%d steps=%d value=%r",
        result.status,
        result.replans,
        result.steps,
        result.value,
    )


if __name__ == "__main__":
    main()
