"""Single source of truth for Longor configuration.

Only secrets and switches worth flipping between runs live in ``.env``
(``OPENAI_API_KEY``, ``LOG_LEVEL``, ``LLM_MODEL``, ``SAVE_VIDEO``). Everything
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
VLA_POLICY = "pi05"  # "pi05" | "smolvla"
VLA_POLICIES = ("pi05", "smolvla")

# --- VLA HTTP service ------------------------------------------------------
VLA_PORT = 8765
VLA_URL = f"http://127.0.0.1:{VLA_PORT}"
VLA_TIMEOUT = 60.0

# --- LIBERO environment ----------------------------------------------------
# 256 is the resolution LIBERO rendered its training demonstrations at; policies
# are sensitive to this, so it is not a free parameter.
RESOLUTION = 256
SEED = 7
# The simulator drops objects onto the table on reset. Hold still, gripper open,
# until they settle -- otherwise the first observation shows objects mid-air.
SETTLE_STEPS = 10
SETTLE_ACTION = [0.0] * 6 + [-1.0]
# Longest training demonstration per suite, rounded up (from the OpenPI eval).
MAX_EPISODE_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# --- video ------------------------------------------------------------------
VIDEO_PATH = "run.mp4"
VIDEO_FPS = 10

# --- VLA manipulation loop --------------------------------------------------
# Max actions returned per /predict response (enforced by the VLA service).
VLA_ACTIONS_PER_CHUNK = 5
# Per-skill step cap (guard against a failing subtask; success path ends earlier).
MAX_SKILL_STEPS = 120
# Guards against a policy returning garbage; the services clip to this too.
ACTION_CLIP = 2.0

# --- planner / executor -----------------------------------------------------
MAX_PLAN_RETRIES = 3
PROGRESS_HISTORY_WINDOW = 3
