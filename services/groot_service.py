"""
GR00T N1.6 HTTP adapter: Longor /predict contract to upstream ZeroMQ policy server.

Bridges multipart PNG + flat state to ``Gr00tSimPolicyWrapper`` flat keys on
``127.0.0.1:5555``. Validates camera/state/action shape and finiteness only.
"""

from __future__ import annotations

import asyncio
import io
import json as _json
import logging
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

import config
from envs.robocasa_env import (
    ACTION_DIMS,
    ACTION_KEYS,
    CAMERA_KEYS,
    IMAGE_SIZE,
    STATE_DIMS,
    STATE_KEYS,
    expected_action_dim,
    expected_state_dim,
    flatten_action_timestep,
    unflatten_state,
    validate_finite_vector,
)
from utils import configure_logging

configure_logging(fmt="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_EXPECTED_IMAGES = len(CAMERA_KEYS)
_IMAGE_SIZE = IMAGE_SIZE
_STATE_DIM = expected_state_dim()
_ACTION_DIM = expected_action_dim()

GROOT_UPSTREAM_HOST = "127.0.0.1"
GROOT_UPSTREAM_PORT = 5555
GROOT_LANGUAGE_KEY = "annotation.human.action.task_description"

_upstream = None


def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode one PNG (or other PIL image) to uint8 RGB at the RoboCasa 256² contract."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def parse_state_field(state: str, expected_dim: int = _STATE_DIM) -> np.ndarray:
    """Parse and validate the flattened RoboCasa proprioceptive state JSON field."""
    if not state:
        raise ValueError(f"state is required ({expected_dim} floats as JSON array)")
    try:
        arr = np.array(_json.loads(state), dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, _json.JSONDecodeError) as exc:
        raise ValueError(f"invalid state JSON: {exc}") from exc
    if arr.size != expected_dim:
        raise ValueError(f"state must have {expected_dim} floats, got {arr.size}")
    validate_finite_vector(arr.tolist(), "state")
    return arr


def truncate_action_chunk(chunk: np.ndarray, max_actions: int) -> np.ndarray:
    """Return at most ``max_actions`` timesteps from a 2D action chunk."""
    if max_actions <= 0:
        raise ValueError(f"max_actions must be positive, got {max_actions}")
    if chunk.ndim != 2:
        raise ValueError(f"expected 2D actions, got shape {chunk.shape}")
    return chunk[:max_actions]


def actions_to_wire_format(chunk: np.ndarray) -> list[list[float]]:
    """GR00T flat RoboCasa vectors per timestep (no rescale or remap)."""
    if chunk.ndim != 2:
        raise ValueError(f"expected 2D actions, got shape {chunk.shape}")
    _t, dim = chunk.shape
    if dim != _ACTION_DIM:
        raise ValueError(f"expected action dim {_ACTION_DIM}, got {dim}")
    out: list[list[float]] = []
    for row in chunk:
        vec = row.astype(np.float32, copy=False).tolist()
        validate_finite_vector(vec, "action")
        out.append(vec)
    return out


def build_groot_observation(
    side0: np.ndarray,
    side1: np.ndarray,
    wrist: np.ndarray,
    task: str,
    state: np.ndarray,
) -> dict[str, object]:
    """Map Longor multipart inputs to Gr00tSimPolicyWrapper flat observation keys."""
    state_groups = unflatten_state(state.tolist())
    obs: dict[str, object] = {
        GROOT_LANGUAGE_KEY: (task,),
    }
    cameras = (side0, side1, wrist)
    for cam_key, rgb in zip(CAMERA_KEYS, cameras):
        batched = rgb.reshape(1, 1, _IMAGE_SIZE, _IMAGE_SIZE, 3)
        obs[f"video.{cam_key}"] = batched
    for key, width in zip(STATE_KEYS, STATE_DIMS):
        vec = np.asarray(state_groups[key], dtype=np.float32).reshape(1, 1, width)
        obs[f"state.{key}"] = vec
    return obs


def groot_action_dict_to_chunk(action_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Convert upstream flat action dict to a (T, action_dim) float32 chunk."""
    horizon = None
    timestep_groups: list[dict[str, list[float]]] = []
    for key, width in zip(ACTION_KEYS, ACTION_DIMS):
        parsed_key = f"action.{key}"
        if parsed_key not in action_dict:
            raise ValueError(f"missing upstream action key {parsed_key!r}")
        arr = np.asarray(action_dict[parsed_key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"{parsed_key} must have shape (B, T, D), got {arr.shape}")
        if arr.shape[0] != 1:
            raise ValueError(f"{parsed_key} batch size must be 1, got {arr.shape[0]}")
        if arr.shape[2] != width:
            raise ValueError(f"{parsed_key} dim must be {width}, got {arr.shape[2]}")
        if horizon is None:
            horizon = int(arr.shape[1])
            timestep_groups = [{} for _ in range(horizon)]
        elif int(arr.shape[1]) != horizon:
            raise ValueError("upstream action horizon mismatch across keys")
        for t in range(horizon):
            timestep_groups[t][key] = [float(v) for v in arr[0, t]]
    if not timestep_groups:
        raise ValueError("upstream returned empty action chunk")
    rows = [flatten_action_timestep(group) for group in timestep_groups]
    return np.asarray(rows, dtype=np.float32)


def _load_upstream_client():
    try:
        from gr00t.policy.server_client import PolicyClient
    except ImportError as exc:
        raise ImportError(
            "Isaac-GR00T is not installed. Run the M2 GR00T server stack on Linux "
            "and install upstream gr00t in the VLA Python environment."
        ) from exc
    client = PolicyClient(
        host=GROOT_UPSTREAM_HOST,
        port=GROOT_UPSTREAM_PORT,
    )
    if not client.ping():
        raise RuntimeError(
            f"upstream GR00T policy server unreachable at "
            f"{GROOT_UPSTREAM_HOST}:{GROOT_UPSTREAM_PORT}"
        )
    return client


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _upstream
    log.info(
        "Connecting GR00T adapter to upstream %s:%d ...",
        GROOT_UPSTREAM_HOST,
        GROOT_UPSTREAM_PORT,
    )
    t0 = time.perf_counter()
    try:
        _upstream = await asyncio.to_thread(_load_upstream_client)
    except ImportError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Upstream connect failed: %s", exc)
        sys.exit(1)
    elapsed = time.perf_counter() - t0
    log.info(
        "GR00T adapter ready in %.1fs — listening on port %d",
        elapsed,
        config.VLA_PORT,
    )
    yield
    _upstream = None


app = FastAPI(title="groot-service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "groot", "action_dim": _ACTION_DIM}


@app.post("/predict")
async def predict(
    images: list[UploadFile] = File(...),
    task: str = Form("Open the drawer"),
    state: str = Form(""),
):
    if _upstream is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(images) != _EXPECTED_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"expected {_EXPECTED_IMAGES} images, got {len(images)}",
        )

    try:
        decoded = []
        for upload in images:
            raw = await upload.read()
            decoded.append(decode_image(raw))
        state_arr = parse_state_field(state)
        obs = build_groot_observation(
            decoded[0], decoded[1], decoded[2], task, state_arr
        )
        action_dict, _info = await asyncio.to_thread(_upstream.get_action, obs)
        chunk = groot_action_dict_to_chunk(action_dict)
        truncated = truncate_action_chunk(chunk, config.VLA_ACTIONS_PER_CHUNK)
        wire = actions_to_wire_format(truncated)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"actions": wire}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.VLA_PORT)
