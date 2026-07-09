"""Pick up a cube in ManiSkill with a handcrafted plan + VLA arm skills.

Requires a running VLA server (see README). Works on Linux or Windows (Git Bash).
"""
import logging
import os

from dotenv import find_dotenv, load_dotenv

from core.plan import AlwaysContinueEvaluator, StaticPlanner, grab_object_plan
from core.runner import Runner
from sim.maniskill_backend import ManiSkillBackend

load_dotenv(find_dotenv())
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
    force=True,
)
for _noisy in ("openai", "httpcore", "httpx", "trimesh", "matplotlib", "numexpr"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

plan = grab_object_plan("cube", goal="pick the cube")
backend = ManiSkillBackend(
    env_id="PickCube-v1",
    render_human=os.environ.get("DISPLAY_ENV", "0") == "1",
)
result = Runner(max_replans=0).run(
    backend,
    plan.goal,
    planner=StaticPlanner(plan),
    feedback=AlwaysContinueEvaluator(),
)
raise SystemExit(0 if result.success else 1)
