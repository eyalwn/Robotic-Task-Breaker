"""M3 — LIBERO-Object with LLMAgent + LeRobot LIBERO backend (Linux)."""
import os
import sys
import time
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import find_dotenv, load_dotenv
import logging

from sim.libero_backend import create_env, LiberoCamera, LiberoArm
from sim.libero_success_checker import LiberoSuccessChecker
from vla import VLAConfig, create_vla_manipulation
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
LIBERO_TASK_SUITE = os.environ.get("LIBERO_TASK_SUITE", "libero_object")
LIBERO_TASK_IDX = int(os.environ.get("LIBERO_TASK_IDX", "0"))
RENDER_HUMAN = os.environ.get("DISPLAY_ENV", "0") == "1"
TASK = os.environ.get("ROBOT_TASK", "complete the manipulation task")
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
os.environ.setdefault("ROBOCREW_MAX_LOOPS", "1")
MAX_LOOPS = int(os.environ.get("ROBOCREW_MAX_LOOPS", "1"))
MAX_SECONDS = float(os.environ.get("ROBOCREW_MAX_SECONDS", "30"))
FRAME_DIR = os.environ.get(
    "ROBOCREW_FRAME_DIR",
    str(os.path.join(os.path.dirname(__file__), "_out")),
)

SYSTEM_PROMPT = (
    "You control a robot arm in simulation (LIBERO). "
    "Follow the task description using the policy tool.\n\n"
    "In this example you must plan in ONE response only.\n"
    "Rules:\n"
    "- In the same response, call exactly two tools in this order:\n"
    "  1) pick_cube_with_policy\n"
    "  2) finish_task\n"
    "- Do not ask for another turn and do not retry pick_cube_with_policy.\n"
    "- finish_task should briefly report the policy result/reason."
)

os.environ.setdefault("VLA_MAX_INNER_STEPS", "60")
os.environ.setdefault("VLA_MAX_TOOL_SECONDS", "90")
os.environ.setdefault("VLA_FAIL_ON_PLACEHOLDER", "1")

env = create_env(
    task_suite=LIBERO_TASK_SUITE,
    task_idx=LIBERO_TASK_IDX,
)
arm = LiberoArm(env)
camera = LiberoCamera(env, arm=arm)

vla_cfg = VLAConfig.from_env()
server_address = f"http://localhost:{VLA_PORT}"
logging.info("Using VLA policy tool at url=%s.", server_address)
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
logging.info(
    "LIBERO suite=%s task_idx=%s",
    LIBERO_TASK_SUITE,
    LIBERO_TASK_IDX,
)
pick_the_cube = create_vla_manipulation(
    tool_name="pick_cube_with_policy",
    tool_description=(
        "Execute a VLA-driven manipulation sequence via the HTTP VLA service."
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
    LiberoSuccessChecker(arm, consecutive_steps=1),
    FrameSaver(FRAME_DIR, disabled=RENDER_HUMAN),
    max_loops=MAX_LOOPS,
    max_seconds=MAX_SECONDS,
)

if RENDER_HUMAN:
    logging.info("Task finished — keeping window open for 1 s...")
    time.sleep(1)
