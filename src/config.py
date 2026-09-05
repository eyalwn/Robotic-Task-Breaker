"""Single source of truth for Longor configuration.

Only secrets and switches worth flipping between runs live in ``.env``. Everything
else is a constant below: to retune the system, edit this file.

``.env`` is loaded here, on import, rather than from each entrypoint. The
constants below are evaluated at import time, so any later ``load_dotenv()``
would come too late to affect them.
"""
from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# --- from .env -------------------------------------------------------------
# OPENAI_API_KEY is not read here; the LLM SDK picks it up from the environment.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip() or "INFO"
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini").strip()
SAVE_VIDEO = os.environ.get("SAVE_VIDEO") == "1"

# --- VLA backend selection (launcher default) ------------------------------
VLA_POLICY = os.environ.get("VLA_POLICY", "pi0fast")  # "pi0fast" | "smolvla"

# --- VLA HTTP service ------------------------------------------------------
VLA_PORT = 8765
VLA_URL = f"http://127.0.0.1:{VLA_PORT}"
VLA_TIMEOUT = int(os.environ.get("VLA_TIMEOUT", "60"))

# --- sim environment -------------------------------------------------------
SIM_ENV = "vlabench"  # "vlabench"

# --- video ------------------------------------------------------------------
VIDEO_PATH = os.environ.get("VIDEO_PATH", "run.mp4")
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", "10"))

# --- VLA manipulation loop --------------------------------------------------
# Max actions returned per /predict response (enforced by the VLA service).
VLA_ACTIONS_PER_CHUNK = int(os.environ.get("VLA_ACTIONS_PER_CHUNK", "5"))
# Per-skill step cap (guard against a failing subtask; success path ends earlier).
MAX_SKILL_STEPS = int(os.environ.get("MAX_SKILL_STEPS", "250"))
# Full-horizon cap for raw VLA baseline episodes (M1 baseline protocol).
MAX_EPISODE_STEPS = 600
# Independent episodes per raw-policy baseline campaign.
BASELINE_EPISODES = 10

# --- planner / executor -----------------------------------------------------
MAX_PLAN_RETRIES = int(os.environ.get("MAX_PLAN_RETRIES", "3"))
