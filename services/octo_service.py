"""
Octo-small-1.5 FastAPI inference service.
Run from repo root: python services/run_vla_server.py (VLA_POLICY=octo-small-1.5 in .env).
Direct: python services/octo_service.py
Env: VLA_MODEL_DIR, VLA_PORT, VLA_ACTION_SCALE
"""

import asyncio
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("octo_service")

def _model_dir_raw() -> str:
    raw = os.environ.get("VLA_MODEL_DIR", str(Path.home() / ".cache" / "octo-small-1.5"))
    return os.path.expanduser(raw.strip())


VLA_MODEL_DIR = _model_dir_raw()
VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
VLA_ACTION_SCALE = float(os.environ.get("VLA_ACTION_SCALE", "0.05"))
VLA_ACTION_CLIP = float(os.environ.get("VLA_ACTION_CLIP", "2.0"))

IMAGE_SIZE = 256
WRIST_SIZE = 128
WINDOW = 1

_model = None
_rng = jax.random.PRNGKey(0)


def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes → (256, 256, 3) uint8 numpy array."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def build_obs(img_np: np.ndarray) -> dict:
    """Build the full observation dict Octo expects from a (256,256,3) uint8 array."""
    return {
        "image_primary": img_np[None, None, ...],  # (1, 1, 256, 256, 3)
        "image_wrist": np.zeros((1, WINDOW, WRIST_SIZE, WRIST_SIZE, 3), dtype=np.uint8),
        "pad_mask_dict": {
            "image_primary": np.ones((1, WINDOW), dtype=bool),
            "image_wrist": np.zeros((1, WINDOW), dtype=bool),
            "timestep": np.ones((1, WINDOW), dtype=bool),
        },
        "task_completed": np.zeros((1, WINDOW, 4), dtype=bool),
        "timestep": np.zeros((1, WINDOW), dtype=np.int32),
        "timestep_pad_mask": np.ones((1, WINDOW), dtype=bool),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    load_arg = VLA_MODEL_DIR
    if not load_arg.startswith("hf://"):
        p = Path(load_arg)
        if not p.is_dir():
            log.error(
                "Model dir not found: %s\n"
                "  Download weights into that folder, or set VLA_MODEL_DIR=hf://rail-berkeley/octo-small-1.5 "
                "(needs network; uses Hugging Face cache).",
                p,
            )
            sys.exit(1)

    log.info("Loading octo-small-1.5 from %s ...", load_arg)
    t0 = time.perf_counter()
    try:
        from octo.model.octo_model import OctoModel
        # load_pretrained calls asyncio.run() internally; must run in a thread
        # so it gets a clean event loop (not the one uvicorn is already running).
        _model = await asyncio.to_thread(OctoModel.load_pretrained, load_arg)
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    log.info("Model loaded in %.1fs — service healthy on port %d", elapsed, VLA_PORT)
    yield
    _model = None


app = FastAPI(title="octo-service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "octo-small-1.5"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    task: str = Form("pick up the cube"),
):
    global _rng

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Decode + resize image
    try:
        raw = await image.read()
        img_np = decode_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}") from exc

    obs = build_obs(img_np)

    # Create task from text
    octo_task = _model.create_tasks(texts=[task])

    # Advance RNG and sample actions
    _rng, sample_rng = jax.random.split(_rng)
    actions = _model.sample_actions(obs, octo_task, rng=sample_rng)
    # actions shape: (batch=1, horizon, action_dim)
    # Return the full horizon so the client can apply the chunk without re-querying.
    action_array = np.array(actions[0, :, :4], dtype=np.float32)
    if action_array.ndim != 2 or action_array.shape[1] != 4:
        raise HTTPException(status_code=500, detail="Model returned invalid action shape")
    clipped_actions = np.clip(action_array, -VLA_ACTION_CLIP, VLA_ACTION_CLIP)
    all_actions = clipped_actions.tolist()

    return {"actions": all_actions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=VLA_PORT)
