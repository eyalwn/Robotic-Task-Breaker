"""
Pi0-FAST (OpenPI) FastAPI inference service for VLABench select_fruit.

Checkpoint: VLABench/pi0-fast-primitive-10task-deltachunk (HF publishes the
same weights as VLABench/pi0-fast-ft-primitive-10task-deltachunk).
Train config: pifast_ft_vlabench_primitive_aligned (delta-chunk actions).

Install deps in .vla-venv: pip install -r deps/requirements-vla-pi0fast-linux.txt
Run: python services/run_vla_server.py
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
from utils import configure_logging

configure_logging(fmt="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PI0FAST_CHECKPOINT = "VLABench/pi0-fast-primitive-10task-deltachunk"
PI0FAST_CHECKPOINT_REVISION = "c58e7d63aa8ef0f6192098eeb5576f771eaf492b"
OPENPI_CONFIG = "pifast_ft_vlabench_primitive_aligned"

_VLABENCH_STATE_DIM = 7
_EXPECTED_IMAGES = 3
_IMAGE_SIZE = 224

# SRD checkpoint id maps to the published Hugging Face repo name.
_HF_CHECKPOINT_ALIASES = {
    "VLABench/pi0-fast-primitive-10task-deltachunk": (
        "VLABench/pi0-fast-ft-primitive-10task-deltachunk"
    ),
}
_INFERENCE_ALLOW_PATTERNS = (
    "_CHECKPOINT_METADATA",
    "assets/**",
    "params/**",
)

_policy = None


def resolve_checkpoint_repo(checkpoint: str) -> str:
    """Map the SRD checkpoint id to the Hugging Face repo that hosts it."""
    return _HF_CHECKPOINT_ALIASES.get(checkpoint, checkpoint)


def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode one JPEG and resize to the OpenPI VLABench visual contract (224)."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def parse_state_field(state: str, expected_dim: int = _VLABENCH_STATE_DIM) -> np.ndarray:
    """Parse and validate the 7-D VLABench proprioceptive state JSON field."""
    if not state:
        raise ValueError(f"state is required ({expected_dim} floats as JSON array)")
    try:
        arr = np.array(_json.loads(state), dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, _json.JSONDecodeError) as exc:
        raise ValueError(f"invalid state JSON: {exc}") from exc
    if arr.size != expected_dim:
        raise ValueError(f"state must have {expected_dim} floats, got {arr.size}")
    return arr


# OpenPI VLABench eval maps ``obs["rgb"][i]`` to these keys in order
# (see Shiduo-zh/openpi VLABench gym wrapper: image_keys below).
PI0FAST_OBSERVATION_IMAGE_KEYS = (
    "observation/image",
    "observation/second_image",
    "observation/wrist_image",
)


def build_pi0fast_observation(
    front_rgb: np.ndarray,
    second_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    task: str,
    state: np.ndarray,
) -> dict:
    """Map front/second/wrist RGB + task into OpenPI VLABench dataset-style keys."""
    return {
        "observation/image": front_rgb,
        "observation/second_image": second_rgb,
        "observation/wrist_image": wrist_rgb,
        "observation/state": state.astype(np.float32, copy=False),
        "prompt": task,
    }


def truncate_action_chunk(chunk: np.ndarray, max_actions: int) -> np.ndarray:
    """Return at most ``max_actions`` timesteps from a 2D action chunk."""
    if max_actions <= 0:
        raise ValueError(f"max_actions must be positive, got {max_actions}")
    if chunk.ndim != 2:
        raise ValueError(f"expected 2D actions, got shape {chunk.shape}")
    return chunk[:max_actions]


def actions_to_wire_format(chunk: np.ndarray) -> list[list[float]]:
    """Pi0-FAST absolute EEF vectors → [x,y,z,rx,ry,rz,grip] per timestep."""
    if chunk.ndim != 2:
        raise ValueError(f"expected 2D actions, got shape {chunk.shape}")
    _t, dim = chunk.shape
    if dim < 7:
        raise ValueError(f"need action dim >= 7 for gripper index 6, got {dim}")
    return chunk[:, :7].astype(np.float32, copy=False).tolist()


def _download_checkpoint(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id,
        revision=PI0FAST_CHECKPOINT_REVISION,
        allow_patterns=list(_INFERENCE_ALLOW_PATTERNS),
        ignore_patterns=["train_state/**"],
    )


def _load_openpi_policy():
    try:
        from openpi.policies import policy_config
        from openpi.training import config as openpi_train_config
    except ImportError as exc:
        raise ImportError(
            "OpenPI (Shiduo-zh fork) is not installed. On Linux: clone "
            "https://github.com/Shiduo-zh/openpi, pip install -e . with the "
            "lerobot git pin from deps/requirements-vla-pi0fast-linux.txt, then "
            "pip install -r deps/requirements-vla-pi0fast-linux.txt in .vla-venv."
        ) from exc

    repo_id = resolve_checkpoint_repo(PI0FAST_CHECKPOINT)
    log.info("Downloading OpenPI checkpoint repo_id=%s ...", repo_id)
    checkpoint_dir = _download_checkpoint(repo_id)
    train_cfg = openpi_train_config.get_config(OPENPI_CONFIG)
    return policy_config.create_trained_policy(train_cfg, checkpoint_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _policy
    log.info(
        "Loading Pi0-FAST (OpenPI) config=%s checkpoint=%s@%s ...",
        OPENPI_CONFIG,
        PI0FAST_CHECKPOINT,
        PI0FAST_CHECKPOINT_REVISION,
    )
    t0 = time.perf_counter()
    try:
        _policy = await asyncio.to_thread(_load_openpi_policy)
    except ImportError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    log.info("Policy loaded in %.1fs — listening on port %d", elapsed, config.VLA_PORT)
    yield
    _policy = None


app = FastAPI(title="pi0fast-service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "pi0fast"}


@app.post("/predict")
async def predict(
    images: list[UploadFile] = File(...),
    task: str = Form("pick up the cube"),
    state: str = Form(""),
):
    if _policy is None:
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
        obs = build_pi0fast_observation(
            decoded[0], decoded[1], decoded[2], task, state_arr
        )
        result = await asyncio.to_thread(_policy.infer, obs)
        actions = result.get("actions")
        if actions is None:
            raise ValueError("Policy returned no 'actions'")
        chunk = np.asarray(actions)
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
