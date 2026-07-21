"""
Pi0.5 (OpenPI) FastAPI inference service — Linux + NVIDIA only.
Run: python services/run_vla_server.py with VLA_POLICY=pi05 in .env.
Direct: python services/pi05_service.py

Env: VLA_PORT, VLA_ACTION_CLIP, OPENPI_CONFIG (train config name, default pi05_droid),
     OPENPI_CHECKPOINT (URI or path; default matches the checkpoint table in OpenPI README).
"""

from __future__ import annotations

import asyncio
import io
import json as _json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pi05_service")

VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
VLA_ACTION_CLIP = float(os.environ.get("VLA_ACTION_CLIP", "2.0"))
OPENPI_CONFIG = os.environ.get("OPENPI_CONFIG", "pi05_droid").strip()
DEFAULT_CHECKPOINTS = {
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
}

IMAGE_SIZE = 224
_policy = None


def decode_image(raw_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def build_openpi_observation(
    config_name: str,
    img_rgb: np.ndarray,
    task: str,
    state: np.ndarray,
    wrist_img_rgb: np.ndarray | None = None,
) -> dict:
    """Map RGB frame(s) + task into OpenPI dataset-style keys for the given config."""
    wrist = wrist_img_rgb if wrist_img_rgb is not None else np.zeros_like(img_rgb)
    key = config_name.lower()
    if "libero" in key:
        return {
            "observation/state": state,
            "observation/image": img_rgb,
            "observation/wrist_image": wrist,
            "prompt": task,
        }
    return {
        "observation/exterior_image_1_left": img_rgb,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": task,
    }


def actions_to_wire_format(chunk: np.ndarray) -> list[list[float]]:
    """Pi0.5 DROID/LIBERO-style vectors → [dx,dy,dz,drx,dry,drz,grip] per timestep."""
    if chunk.ndim != 2:
        raise ValueError(f"expected 2D actions, got shape {chunk.shape}")
    _t, dim = chunk.shape
    if dim < 7:
        raise ValueError(f"need action dim >= 7 for gripper index 6, got {dim}")
    merged = chunk[:, :7].astype(np.float32, copy=False)
    clipped = np.clip(merged, -VLA_ACTION_CLIP, VLA_ACTION_CLIP)
    return clipped.tolist()


def _default_checkpoint_uri(config_name: str) -> str:
    if config_name not in DEFAULT_CHECKPOINTS:
        known = ", ".join(sorted(DEFAULT_CHECKPOINTS))
        raise ValueError(
            f"No default checkpoint for OPENPI_CONFIG={config_name!r}; set OPENPI_CHECKPOINT. "
            f"Known defaults: {known}."
        )
    return DEFAULT_CHECKPOINTS[config_name]


def _load_openpi_policy():
    try:
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_train_config
    except ImportError as exc:
        raise ImportError(
            "OpenPI is not installed or its dependencies failed to import. "
            "On Linux: clone https://github.com/Physical-Intelligence/openpi, "
            "run `uv sync` / `pip install -e .` per upstream docs, then "
            "`pip install -r deps/requirements-vla-pi05-linux.txt` in the same venv."
        ) from exc

    ckpt = (os.environ.get("OPENPI_CHECKPOINT") or "").strip() or _default_checkpoint_uri(
        OPENPI_CONFIG
    )
    train_cfg = openpi_train_config.get_config(OPENPI_CONFIG)
    checkpoint_dir = download.maybe_download(ckpt)
    return policy_config.create_trained_policy(train_cfg, checkpoint_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _policy
    log.info("Loading Pi0.5 (OpenPI) config=%s ...", OPENPI_CONFIG)
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
    log.info("Policy loaded in %.1fs — listening on port %d", elapsed, VLA_PORT)
    yield
    _policy = None


app = FastAPI(title="pi05-service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "pi05"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    image_wrist: UploadFile | None = File(None),
    task: str = Form("pick up the cube"),
    state: str = Form(""),
):
    if _policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw = await image.read()
        img_np = decode_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}") from exc

    wrist_np = None
    if image_wrist is not None:
        try:
            raw_wrist = await image_wrist.read()
            if raw_wrist:
                wrist_np = decode_image(raw_wrist)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Wrist image decode error: {exc}"
            ) from exc

    if state:
        try:
            state_arr = np.array(_json.loads(state), dtype=np.float32)
        except (ValueError, TypeError):
            log.warning("Invalid state field; using zeros")
            state_arr = np.zeros(8, dtype=np.float32)
    else:
        state_arr = np.zeros(8, dtype=np.float32)

    obs = build_openpi_observation(
        OPENPI_CONFIG, img_np, task, state_arr, wrist_img_rgb=wrist_np
    )

    try:
        result = await asyncio.to_thread(_policy.infer, obs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    actions = result.get("actions")
    if actions is None:
        raise HTTPException(status_code=500, detail="Policy returned no 'actions'")
    chunk = np.asarray(actions)
    try:
        wire = actions_to_wire_format(chunk)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"actions": wire}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=VLA_PORT)
