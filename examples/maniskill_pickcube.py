"""Pick up a cube in ManiSkill with a handcrafted plan + VLA arm skills.

Requires a running VLA server (see README). Works on Linux or Windows (Git Bash).
"""
import os

from core.plan import AlwaysContinueEvaluator, StaticPlanner, grab_object_plan
from core.runner import Runner
from sim.maniskill_backend import ManiSkillBackend
from utils import configure_logging, load_env

load_env()
configure_logging()

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
