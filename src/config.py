"""Single source of truth for Longor configuration.

Only secrets and switches worth flipping between runs live in ``.env``. Everything
else is a constant below: to retune the system, edit this file.

``.env`` is loaded here, on import, rather than from each entrypoint. The
constants below are evaluated at import time, so any later ``load_dotenv()``
would come too late to affect them.
"""
from __future__ import annotations

import ast
import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# --- App level -------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip() or "INFO"
# Full-horizon cap for raw VLA baseline episodes (M1 baseline protocol).
MAX_EPISODE_STEPS = 600
# Independent episodes per raw-policy baseline campaign.
BASELINE_EPISODES = 10

# --- video ------------------------------------------------------------------
SAVE_VIDEO = os.environ.get("SAVE_VIDEO") == "1"
VIDEO_PATH = os.environ.get("VIDEO_PATH", "run.mp4")
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", "10"))

# --- sim environment -------------------------------------------------------
SIM_ENV = "vlabench"  # "vlabench"

# --- Planner ---------------------------------------------------------------
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini").strip()
MAX_PLAN_RETRIES = int(os.environ.get("MAX_PLAN_RETRIES", "3"))
# Extra observe→plan→execute cycles after skill abort or subtask timeout.
MAX_REPLANS = 2

# --- Executor --------------------------------------------------------------
# Per-skill step cap (guard against a failing subtask; success path ends earlier).
MAX_SKILL_STEPS = ast.literal_eval(
    os.environ.get("MAX_SKILL_STEPS", '{"transfer": 250}')
)
# temporary for testing plate_seen hypothesis
FORCE_TARGET_NAME = os.environ.get("FORCE_TARGET_NAME", "")

# Transfer GT abort (sim-oracle): stall patience, hold/drop distances, progress eps.
SKILL_STALL_STEPS = 20
TRANSFER_HOLD_DISTANCE = 0.12
TRANSFER_DROP_DISTANCE = 0.25
TRANSFER_PROGRESS_EPS = 0.01

# --- VLA server ------------------------------------------------------------
# VLA backend selection (launcher default)
VLA_POLICY = os.environ.get("VLA_POLICY", "pi0fast")  # "pi0fast" | "smolvla"
# Max actions returned per /predict response (enforced by the VLA service).
VLA_ACTIONS_PER_CHUNK = int(os.environ.get("VLA_ACTIONS_PER_CHUNK", "5"))
VLA_PORT = 8765
VLA_URL = f"http://127.0.0.1:{VLA_PORT}"
VLA_TIMEOUT = int(os.environ.get("VLA_TIMEOUT", "60"))
