"""M5 — LIBERO-Object with handcrafted plan + sim-GT skills (Linux)."""
import logging
import os
import time

from dotenv import find_dotenv, load_dotenv

from core.plan import AlwaysContinueEvaluator, StaticPlanner, validate_plan
from core.runner import Runner
from sim.libero_backend import LiberoBackend

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

_OBJECT_NAME = "alphabet_soup_1"
_FIXTURE_PARTS = ("basket", "cabinet", "plate", "drawer", "bin")


def _resolve_object_name(default: str, objects: list[str]) -> str:
    if default in objects or not objects:
        return default
    for name in sorted(objects):
        low = name.lower()
        if any(part in low for part in _FIXTURE_PARTS):
            continue
        return name
    return default


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
    backend = LiberoBackend(task_suite=LIBERO_TASK_SUITE, task_idx=LIBERO_TASK_IDX)
    objects = backend.list_objects()
    logging.info("LIBERO scene objects: %s", objects)
    object_name = _resolve_object_name(_OBJECT_NAME, objects)
    if object_name != _OBJECT_NAME:
        logging.info("Using discovered object name: %s", object_name)
    plan = _build_handcrafted_plan(object_name)
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
