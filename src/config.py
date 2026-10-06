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

# --- App level -------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip() or "INFO"

# --- video ------------------------------------------------------------------
SAVE_VIDEO = os.environ.get("SAVE_VIDEO") == "1"
VIDEO_PATH = os.environ.get("VIDEO_PATH", "run.mp4")
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", "10"))

# --- sim environment -------------------------------------------------------
SIM_ENV = "robocasa"  # "robocasa"

# --- Planner ---------------------------------------------------------------
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini").strip()
MAX_PLAN_RETRIES = int(os.environ.get("MAX_PLAN_RETRIES", "3"))
# Extra observe→plan→execute cycles after subtask timeout.
MAX_REPLANS = 2

# --- Executor --------------------------------------------------------------
# Per-skill step cap (guard against a failing subtask; success path ends earlier).
MAX_SKILL_STEPS = {"open": 720, "place": 900, "turn_on": 400}

# --- VLA server ------------------------------------------------------------
# VLA backend selection (launcher default)
VLA_POLICY = os.environ.get("VLA_POLICY", "groot")  # "groot"
# Actions per /predict response (GR00T N1.6 default chunk size; service enforces
# cap).
VLA_ACTIONS_PER_CHUNK = int(os.environ.get("VLA_ACTIONS_PER_CHUNK", "8"))
VLA_PORT = 8765
VLA_URL = f"http://127.0.0.1:{VLA_PORT}"
VLA_TIMEOUT = int(os.environ.get("VLA_TIMEOUT", "60"))
