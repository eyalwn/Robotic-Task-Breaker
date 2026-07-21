"""
SmolVLA (LeRobot) FastAPI inference service — Linux + NVIDIA (T4-class OK).
Run: python services/run_vla_server.py with VLA_POLICY=smolvla in .env.
Direct: python services/smolvla_service.py

Env: VLA_PORT, VLA_ACTION_CLIP, SMOLVLA_CHECKPOINT (default HuggingFaceVLA/smolvla_libero).
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
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smolvla_service")

VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
VLA_ACTION_CLIP = float(os.environ.get("VLA_ACTION_CLIP", "2.0"))
SMOLVLA_CHECKPOINT = os.environ.get(
    "SMOLVLA_CHECKPOINT", "HuggingFaceVLA/smolvla_libero"
).strip()

IMAGE_SIZE = 256

_policy_bundle: "PolicyBundle | None" = None


@dataclass
class PolicyBundle:
    policy: object
    preprocessor: object
    postprocessor: object
    device: object


def decode_image(raw_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def parse_state_field(state: str) -> np.ndarray:
    if not state:
        return np.zeros(8, dtype=np.float32)
    try:
        arr = np.array(_json.loads(state), dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, _json.JSONDecodeError):
        log.warning("Invalid state field; using zeros")
        return np.zeros(8, dtype=np.float32)
    if arr.size != 8:
        log.warning("State dim %d != 8; using zeros", arr.size)
        return np.zeros(8, dtype=np.float32)
    return arr


def build_smolvla_observation(
    img_rgb: np.ndarray,
    task: str,
    state: np.ndarray,
    wrist_img_rgb: np.ndarray | None = None,
) -> dict:
    """Map RGB frame(s) + task into LeRobot SmolVLA LIBERO keys."""
    wrist = wrist_img_rgb if wrist_img_rgb is not None else np.zeros_like(img_rgb)
    return {
        "observation.images.image": img_rgb,
        "observation.images.image2": wrist,
        "observation.state": state.astype(np.float32, copy=False),
        "task": task,
    }


def actions_to_wire_format(chunk: np.ndarray) -> list[list[float]]:
    """SmolVLA LIBERO vectors → [dx,dy,dz,drx,dry,drz,grip] per timestep."""
    if chunk.ndim != 2:
        raise ValueError(f"expected 2D actions, got shape {chunk.shape}")
    _t, dim = chunk.shape
    if dim < 7:
        raise ValueError(f"need action dim >= 7 for gripper index 6, got {dim}")
    merged = chunk[:, :7].astype(np.float32, copy=False)
    clipped = np.clip(merged, -VLA_ACTION_CLIP, VLA_ACTION_CLIP)
    return clipped.tolist()


def _import_smolvla():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. On Linux + NVIDIA (T4-class Colab): "
            "pip install -r deps/requirements-vla-smolvla-linux.txt in .vla-venv."
        ) from exc

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        try:
            from lerobot.policies.smolvla import SmolVLAPolicy
        except ImportError as exc:
            raise ImportError(
                "LeRobot SmolVLA is not installed. On Linux: "
                "pip install -r deps/requirements-vla-smolvla-linux.txt "
                "in the same .vla-venv used for services/run_vla_server.py."
            ) from exc

    try:
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError:
        from lerobot.policies import make_pre_post_processors

    return torch, SmolVLAPolicy, make_pre_post_processors


def _load_smolvla_policy() -> PolicyBundle:
    torch, SmolVLAPolicy, make_pre_post_processors = _import_smolvla()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading SmolVLA checkpoint=%s on %s ...", SMOLVLA_CHECKPOINT, device)

    policy = SmolVLAPolicy.from_pretrained(SMOLVLA_CHECKPOINT)
    policy.to(device)
    policy.eval()

    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=SMOLVLA_CHECKPOINT,
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return PolicyBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
    )


def _prepare_observation_for_inference(
    observation: dict,
    device: object,
    task: str = "",
) -> dict:
    """HWC uint8 numpy → BCHW float32 tensors (LeRobot inference contract)."""
    import torch

    prepared = {}
    for name, value in observation.items():
        tensor = torch.from_numpy(value)
        if "image" in name:
            if tensor.dtype == torch.uint8:
                tensor = tensor.type(torch.float32) / 255
            tensor = tensor.permute(2, 0, 1).contiguous()
        tensor = tensor.unsqueeze(0).to(device)
        prepared[name] = tensor
    prepared["task"] = task
    prepared["robot_type"] = ""
    return prepared


def _run_inference(bundle: PolicyBundle, obs: dict) -> np.ndarray:
    import torch

    bundle.policy.reset()
    task = obs.pop("task", "")
    obs = _prepare_observation_for_inference(obs, bundle.device, task=task)
    batch = bundle.preprocessor(obs)
    chunk = bundle.policy.predict_action_chunk(batch)
    if not isinstance(chunk, torch.Tensor):
        raise ValueError(f"policy returned {type(chunk)}, expected torch.Tensor")
    if chunk.ndim == 2:
        chunk = chunk.unsqueeze(0)
    if chunk.ndim != 3:
        raise ValueError(f"expected action chunk ndim 3, got shape {tuple(chunk.shape)}")
    if chunk.shape[0] != 1:
        raise ValueError(f"expected batch size 1, got shape {tuple(chunk.shape)}")

    _, chunk_size, action_dim = chunk.shape
    if action_dim < 7:
        raise ValueError(f"need action dim >= 7, got {action_dim}")

    processed = []
    for idx in range(chunk_size):
        processed.append(bundle.postprocessor(chunk[:, idx, :]))
    stacked = torch.stack(processed, dim=1).squeeze(0)
    return stacked.detach().cpu().numpy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _policy_bundle
    log.info("Loading SmolVLA checkpoint=%s ...", SMOLVLA_CHECKPOINT)
    t0 = time.perf_counter()
    try:
        _policy_bundle = await asyncio.to_thread(_load_smolvla_policy)
    except ImportError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    log.info("Policy loaded in %.1fs — listening on port %d", elapsed, VLA_PORT)
    yield
    _policy_bundle = None


app = FastAPI(title="smolvla-service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "smolvla"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    image_wrist: UploadFile | None = File(None),
    task: str = Form("pick up the cube"),
    state: str = Form(""),
):
    if _policy_bundle is None:
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

    state_arr = parse_state_field(state)
    obs = build_smolvla_observation(img_np, task, state_arr, wrist_img_rgb=wrist_np)

    try:
        chunk_np = await asyncio.to_thread(_run_inference, _policy_bundle, obs)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    try:
        wire = actions_to_wire_format(chunk_np)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"actions": wire}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=VLA_PORT)
