"""M5 — ManiSkill PickCube with handcrafted plan + sim-GT skills (Windows smoke test)."""
import logging
import os
import sys
import time
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import find_dotenv, load_dotenv

from core.plan import AlwaysContinueEvaluator, StaticPlanner, validate_plan
from core.runner import Runner
from sim.maniskill_backend import ManiSkillBackend

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

RENDER_HUMAN = os.environ.get("DISPLAY_ENV", "0") == "1"
OBJECT_NAME = "cube"
GOAL = "pick the cube"


def main() -> None:
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

    backend = ManiSkillBackend(render_human=RENDER_HUMAN)
    result = Runner(max_replans=0).run(
        backend,
        plan.goal,
        planner=StaticPlanner(plan),
        feedback=AlwaysContinueEvaluator(),
    )
    logging.info(
        "Runner result: status=%s success=%s replans=%d steps=%d value=%r",
        result.status,
        result.success,
        result.replans,
        result.steps,
        result.value,
    )

    if RENDER_HUMAN:
        logging.info("Task finished — keeping window open for 1 s...")
        time.sleep(1)


if __name__ == "__main__":
    main()
