"""M5 — ManiSkill PickCube with handcrafted plan + sim-GT skills (Windows smoke test)."""
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
from sim.maniskill_backend import ManiSkillArm, ManiSkillCamera, create_env
from sim.maniskill_skills import register_maniskill_skills
from sim.success_checker import PickCubeSuccessChecker
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

ENV_ID = os.environ.get("MANISKILL_ENV", "PickCube-v1")
RENDER_HUMAN = os.environ.get("DISPLAY_ENV", "0") == "1"
TASK = os.environ.get("ROBOT_TASK", "pick the cube")
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
OBJECT_NAME = "cube"
GOAL = "pick the cube"

os.environ.setdefault("VLA_AXIS_REMAP", "2,0,1")
os.environ.setdefault("VLA_AXIS_SIGNS", "1,-1,-1")
os.environ.setdefault("VLA_SIM_PICKCUBE_GRIPPER_PROFILE", "true")


class _AlwaysContinue:
    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackDecision:
        logging.info("Feedback [%s] → CONTINUE (handcrafted stub)", description)
        return FeedbackDecision.CONTINUE


class _NoReplan:
    def plan(self, goal, catalog, context=""):
        raise RuntimeError("Replanning not available in handcrafted-plan mode")


def main() -> None:
    env = None
    try:
        env = create_env(ENV_ID, render_human=RENDER_HUMAN)
        arm = ManiSkillArm(env)
        camera = ManiSkillCamera(env, arm=arm)
        success_checker = PickCubeSuccessChecker(env, consecutive_steps=1)

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
        register_maniskill_skills(registry, env)
        register_arm_skills(registry, vla_callable)

        list_skill = registry.get("state.list_objects")
        objects = list_skill()
        logging.info("ManiSkill scene objects: %s", objects)

        plan = validate_plan(
            {
                "goal": GOAL,
                "body": {
                    "kind": "seq",
                    "steps": [
                        {
                            "kind": "call",
                            "skill": "perception.locate",
                            "args": {
                                "object_name": {
                                    "kind": "literal",
                                    "value": OBJECT_NAME,
                                }
                            },
                            "assign_to": "obj_pose",
                        },
                        {
                            "kind": "feedback",
                            "description": (
                                f"Is {OBJECT_NAME} visible and reachable?"
                            ),
                        },
                        {
                            "kind": "call",
                            "skill": "arm.grab",
                            "args": {
                                "object_name": {
                                    "kind": "literal",
                                    "value": OBJECT_NAME,
                                }
                            },
                            "assign_to": "grab_result",
                        },
                        {
                            "kind": "feedback",
                            "description": f"Is {OBJECT_NAME} grasped?",
                        },
                        {
                            "kind": "return",
                            "value": {"kind": "var", "name": "grab_result"},
                        },
                    ],
                },
            }
        )

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
        logging.info(
            "PickCube success (diagnostic only): %s",
            success_checker.check(),
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
