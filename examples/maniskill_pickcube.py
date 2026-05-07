"""Milestone 2 — PickCube with LLMAgent + ManiSkill sim backend.

Observation/render sanity for VLA:
- Keep DISPLAY_ENV=0 for policy runs (camera uses rgb_array render path).
- DISPLAY_ENV=1 is only for live viewing and may use fallback capture paths.
- If Vulkan reports DeviceLost/ErrorDeviceLost, policy vision is unusable for this run.
  Fix rendering first (driver/apps/headless mode) before tuning VLA axis settings.
"""
import os
import sys
import time
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import find_dotenv, load_dotenv
import logging

from sim.maniskill_backend import create_env, ManiSkillCamera, ManiSkillArm
from vla import VLAConfig, create_vla_manipulation
from sim.success_checker import PickCubeSuccessChecker
from sim.frame_saver import FrameSaver
from sim.runner import run_sim_agent
from core.tools import finish_task
from core.LLMAgent import LLMAgent

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

MODEL = os.environ.get("LLM_MODEL")
ENV_ID = os.environ.get("MANISKILL_ENV", "PickCube-v1")
RENDER_HUMAN = os.environ.get("DISPLAY_ENV", "0") == "1"
TASK = os.environ.get("ROBOT_TASK", "pick the cube")
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
MAX_LOOPS = int(os.environ.get("ROBOCREW_MAX_LOOPS", "1"))
MAX_SECONDS = float(os.environ.get("ROBOCREW_MAX_SECONDS", "30"))
FRAME_DIR = os.environ.get(
    "ROBOCREW_FRAME_DIR",
    str(os.path.join(os.path.dirname(__file__), "_out")),
)
# Debug note:
# ROBOCREW_MAX_SECONDS is a between-turn budget (checked before each LLM turn), not
# a strict wall-clock cap for one long-running tool call. A single
# pick_cube_with_policy call may exceed it on slow machines. For local debugging, try:
#   ROBOCREW_MAX_SECONDS=120
#   ROBOCREW_MAX_LOOPS=3
# Keep CI/default runs on the conservative defaults above.

SYSTEM_PROMPT = (
    "You control a robot arm in simulation. Your goal is to pick the cube.\n\n"
    "In this example you must plan in ONE response only.\n"
    "Rules:\n"
    "- In the same response, call exactly two tools in this order:\n"
    "  1) pick_cube_with_policy\n"
    "  2) finish_task\n"
    "- Do not ask for another turn and do not retry pick_cube_with_policy.\n"
    "- finish_task should briefly report the policy result/reason."
)

# Default to the known-good CPU sim calibration from recent PickCube runs.
# Users can still override these in their environment.
os.environ.setdefault("VLA_AXIS_REMAP", "2,0,1")
os.environ.setdefault("VLA_AXIS_SIGNS", "1,-1,-1")
os.environ.setdefault("VLA_SIM_PICKCUBE_GRIPPER_PROFILE", "true")

env = create_env(ENV_ID, render_human=RENDER_HUMAN)
arm = ManiSkillArm(env)
camera = ManiSkillCamera(env, arm=arm)

vla_cfg = VLAConfig.from_env()
server_address = f"http://localhost:{VLA_PORT}"
logging.info("Using Octo VLA policy tool at url=%s.", server_address)
logging.info(
    "Run turn budget: max_loops=%d max_seconds=%.1f "
    "(checked between turns; env: ROBOCREW_MAX_LOOPS/ROBOCREW_MAX_SECONDS).",
    MAX_LOOPS,
    MAX_SECONDS,
)
logging.info(
    "VLA inner budget: max_inner_steps=%d max_tool_seconds=%s",
    vla_cfg.max_inner_steps,
    vla_cfg.max_tool_seconds,
)
pick_the_cube = create_vla_manipulation(
    tool_name="pick_cube_with_policy",
    tool_description=(
        "Execute a VLA-driven pick sequence by querying the Octo service."
    ),
    task_prompt=TASK,
    server_address=server_address,
    arm=arm,
    camera=camera,
    cfg=vla_cfg,
)

tools = [pick_the_cube, finish_task]

agent = LLMAgent(model=MODEL, tools=tools, main_camera=camera,
                 system_prompt=SYSTEM_PROMPT, history_len=6)
agent.task = TASK

summary = run_sim_agent(
    agent,
    camera,
    PickCubeSuccessChecker(env, consecutive_steps=1),
    FrameSaver(FRAME_DIR, disabled=RENDER_HUMAN),
    max_loops=MAX_LOOPS,
    max_seconds=MAX_SECONDS,
)

if RENDER_HUMAN:
    logging.info("Task finished — keeping window open for 1 s...")
    time.sleep(1)
