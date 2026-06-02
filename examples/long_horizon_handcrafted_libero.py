"""M5 — LIBERO-Object with handcrafted plan + sim-GT skills (Linux)."""
import os
import sys
import time
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import find_dotenv, load_dotenv
import logging

from core.plan import Executor, validate_plan
from core.plan.feedback import FeedbackDecision
from core.skills import SkillRegistry
from sim.libero_backend import LiberoArm, LiberoCamera, create_env
from sim.libero_skills import register_libero_skills
from vla import VLAConfig, create_vla_manipulation
from vla.arm_skills import register_arm_skills

load_dotenv(find_dotenv())
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
    force=True,
)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("trimesh").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

LIBERO_TASK_SUITE = os.environ.get("LIBERO_TASK_SUITE", "libero_object")
LIBERO_TASK_IDX = int(os.environ.get("LIBERO_TASK_IDX", "0"))
RENDER_HUMAN = os.environ.get("DISPLAY_ENV", "0") == "1"
TASK = os.environ.get("ROBOT_TASK", "complete the manipulation task")
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))

# libero_object task 0: pick_up_the_alphabet_soup_and_place_it_in_the_basket
_OBJECT_NAME = "alphabet_soup_1"


class _AlwaysContinue:
    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackDecision:
        logging.info("Feedback [%s] → CONTINUE (handcrafted stub)", description)
        return FeedbackDecision.CONTINUE


class _NoReplan:
    def plan(self, goal, catalog, context=""):
        raise RuntimeError("Replanning not available in handcrafted-plan mode")


def _build_handcrafted_plan(object_name: str):
    return validate_plan(
        {
            "goal": f"pick up {object_name}",
            "body": {
                "kind": "seq",
                "steps": [
                    {
                        "kind": "call",
                        "skill": "perception.locate",
                        "args": {
                            "object_name": {
                                "kind": "literal",
                                "value": object_name,
                            }
                        },
                        "assign_to": "obj_pose",
                    },
                    {
                        "kind": "feedback",
                        "description": (
                            f"Is {object_name} visible and reachable?"
                        ),
                    },
                    {
                        "kind": "call",
                        "skill": "arm.grab",
                        "args": {
                            "object_name": {
                                "kind": "literal",
                                "value": object_name,
                            }
                        },
                        "assign_to": "grab_result",
                    },
                    {
                        "kind": "feedback",
                        "description": f"Is {object_name} grasped?",
                    },
                    {
                        "kind": "return",
                        "value": {"kind": "var", "name": "grab_result"},
                    },
                ],
            },
        }
    )


def main() -> None:
    env = None
    try:
        env = create_env(
            task_suite=LIBERO_TASK_SUITE,
            task_idx=LIBERO_TASK_IDX,
        )
        arm = LiberoArm(env)
        camera = LiberoCamera(env, arm=arm)

        vla_cfg = VLAConfig.from_env()
        vla_lc_tool = create_vla_manipulation(
            tool_name="arm_vla",
            tool_description="VLA-driven arm manipulation",
            task_prompt=TASK,
            server_address=f"http://localhost:{VLA_PORT}",
            arm=arm,
            camera=camera,
            cfg=vla_cfg,
        )
        vla_callable = lambda task: vla_lc_tool.invoke({"task": task})

        registry = SkillRegistry()
        register_libero_skills(registry, env)
        register_arm_skills(registry, vla_callable)

        list_skill = registry.get("state.list_objects")
        objects = list_skill()
        logging.info("LIBERO scene objects: %s", objects)

        object_name = _OBJECT_NAME
        if object_name not in objects and objects:
            skip = ("basket", "cabinet", "plate", "drawer", "bin")
            for name in sorted(objects):
                low = name.lower()
                if any(part in low for part in skip):
                    continue
                object_name = name
                break
            logging.info("Using discovered object name: %s", object_name)

        plan = _build_handcrafted_plan(object_name)

        executor = Executor(
            registry=registry,
            planner=_NoReplan(),
            feedback_evaluator=_AlwaysContinue(),
            image_fn=camera.capture_image,
            max_replans=0,
            step_budget=30,
        )
        result = executor.run(goal=plan.goal, plan=plan)
        logging.info(
            "Executor result: status=%s replans=%d steps=%d value=%r",
            result.status,
            result.replans,
            result.steps,
            result.value,
        )
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception as exc:
                logging.debug("env.close() failed: %s", exc)

    if RENDER_HUMAN:
        logging.info("Task finished — keeping window open for 1 s...")
        time.sleep(1)


if __name__ == "__main__":
    main()
