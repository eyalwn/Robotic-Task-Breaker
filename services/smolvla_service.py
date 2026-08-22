"""
SmolVLA (LeRobot) FastAPI inference service — Linux + NVIDIA (T4-class OK).
Default VLA backend via services/run_vla_server.py.
Install deps in .vla-venv: pip install -r deps/requirements-vla-smolvla-linux.txt
Run: python services/run_vla_server.py

Env: VLA_PORT, VLA_ACTION_CLIP, SMOLVLA_CHECKPOINT (default lerobot/smolvla_vlabench).
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

import config
from utils import configure_logging, load_env

load_env()
configure_logging(fmt="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
VLA_ACTION_CLIP = float(os.environ.get("VLA_ACTION_CLIP", "2.0"))
SMOLVLA_CHECKPOINT = os.environ.get(
    "SMOLVLA_CHECKPOINT", "lerobot/smolvla_vlabench"
).strip()

IMAGE_SIZE = 256

_policy_bundle: PolicyBundle = False  # type: ignore[assignment]


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
        raise ValueError("state is required (7 floats as JSON array)")
    try:
        arr = np.array(_json.loads(state), dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, _json.JSONDecodeError) as exc:
        raise ValueError(f"invalid state JSON: {exc}") from exc
    if arr.size != 7:
        raise ValueError(f"state must have 7 floats, got {arr.size}")
    return arr


def build_smolvla_observation(
    front_rgb: np.ndarray,
    second_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    task: str,
    state: np.ndarray,
) -> dict:
    """Map RGB frames + task into LeRobot SmolVLA VLAbench keys."""
    return {
        "observation.images.camera1": front_rgb,
        "observation.images.camera2": second_rgb,
        "observation.images.camera3": wrist_rgb,
        "observation.state": state.astype(np.float32, copy=False),
        "task": task,
    }


def truncate_action_chunk(chunk: np.ndarray, max_actions: int) -> np.ndarray:
    """Return at most ``max_actions`` timesteps from a 2D action chunk."""
    if max_actions <= 0:
        raise ValueError(f"max_actions must be positive, got {max_actions}")
    if chunk.ndim != 2:
        raise ValueError(f"expected 2D actions, got shape {chunk.shape}")
    return chunk[:max_actions]


def actions_to_wire_format(chunk: np.ndarray) -> list[list[float]]:
    """SmolVLA vectors → [dx,dy,dz,drx,dry,drz,grip] per timestep."""
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
                "in .vla-venv, then run: python services/run_vla_server.py --policy smolvla "
                "(or python services/smolvla_service.py directly)."
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
        if "image" in name or "camera" in name:
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
    _policy_bundle = False


app = FastAPI(title="smolvla-service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "smolvla"}


@app.post("/predict")
async def predict(
    images: list[UploadFile] = File(...),
    task: str = Form("pick up the cube"),
    state: str = Form(""),
):
    if _policy_bundle is False:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(images) != 3:
        raise HTTPException(
            status_code=400,
            detail=f"expected 3 images, got {len(images)}",
        )

    try:
        decoded = []
        for upload in images:
            raw = await upload.read()
            decoded.append(decode_image(raw))
        state_arr = parse_state_field(state)
        obs = build_smolvla_observation(
            decoded[0], decoded[1], decoded[2], task, state_arr
        )
        chunk_np = await asyncio.to_thread(_run_inference, _policy_bundle, obs)
        truncated = truncate_action_chunk(chunk_np, config.VLA_ACTIONS_PER_CHUNK)
        wire = actions_to_wire_format(truncated)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"actions": wire}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=VLA_PORT)
