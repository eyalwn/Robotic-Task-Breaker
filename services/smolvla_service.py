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
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

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
_first_predict_debug_logged = False

def _inactivity_timeout_sec() -> float:
    return float(os.environ.get("SMOLVLA_INACTIVITY_TIMEOUT_SEC", "60"))


def _episode_boundary(task: str, now_mono: float) -> bool:
    state = _action_queue_state
    if state.last_task and task != state.last_task:
        return True
    timeout = _inactivity_timeout_sec()
    if (
        state.last_activity_mono > 0
        and timeout > 0
        and now_mono - state.last_activity_mono >= timeout
    ):
        return True
    return False


@dataclass
class _ActionQueueState:
    """Server-side action queue mirroring lerobot-eval select_action buffering."""

    pending: deque[list[float]] = field(default_factory=deque)
    last_task: str = ""
    last_activity_mono: float = 0.0


_action_queue_state = _ActionQueueState()


def _smolvla_debug_first_predict_enabled() -> bool:
    raw = os.environ.get("SMOLVLA_DEBUG_FIRST_PREDICT", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _maybe_log_first_predict_debug(state_arr: np.ndarray, chunk_np: np.ndarray) -> None:
    global _first_predict_debug_logged
    if _first_predict_debug_logged or not _smolvla_debug_first_predict_enabled():
        return
    _first_predict_debug_logged = True
    log.info(
        "SmolVLA first predict state min=%.4f max=%.4f mean=%.4f",
        float(state_arr.min()),
        float(state_arr.max()),
        float(state_arr.mean()),
    )
    log.info(
        "SmolVLA first predict action chunk min=%.4f max=%.4f mean=%.4f",
        float(chunk_np.min()),
        float(chunk_np.max()),
        float(chunk_np.mean()),
    )


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


def _serve_batch_size() -> int | None:
    """Actions to return per /predict; None = drain entire pending queue."""
    override = os.environ.get("SMOLVLA_ACTIONS_PER_PREDICT", "").strip()
    if override:
        value = int(override)
        return None if value <= 0 else value
    raw = os.environ.get("VLA_CHUNK_STEPS", "").strip()
    if not raw or raw == "0":
        return None
    return int(raw)


def _reset_episode(bundle: PolicyBundle) -> None:
    bundle.policy.reset()
    _action_queue_state.pending.clear()
    log.info("SmolVLA episode reset (task/inactivity boundary)")


def _clear_action_queue_state() -> None:
    """Reset queue/session state (tests and cold-start)."""
    _action_queue_state.pending.clear()
    _action_queue_state.last_task = ""
    _action_queue_state.last_activity_mono = 0.0


def _pop_pending_actions(batch_size: int | None) -> list[list[float]]:
    state = _action_queue_state
    if not state.pending:
        return []
    if batch_size is None:
        actions = list(state.pending)
        state.pending.clear()
        return actions
    actions: list[list[float]] = []
    for _ in range(min(batch_size, len(state.pending))):
        actions.append(state.pending.popleft())
    return actions


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
            "pip install -r services/requirements-vla-smolvla-linux.txt in .vla-venv."
        ) from exc

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        try:
            from lerobot.policies.smolvla import SmolVLAPolicy
        except ImportError as exc:
            raise ImportError(
                "LeRobot SmolVLA is not installed. On Linux: "
                "pip install -r services/requirements-vla-smolvla-linux.txt "
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
    """HWC uint8 numpy → BCHW float32 tensors for LeRobot SmolVLA inference.

    Contract (matches HuggingFaceVLA/smolvla_libero + make_pre_post_processors):
    - ``observation.images.*``: uint8 HWC → float32 BCHW in [0,1] before preprocessor
      (preprocessor may apply dataset mean/std on top).
    - ``observation.state``: float32 (8,) proprio — layout from client (see
      LIBERO_VLA_STATE_MODE on the sim backend).
    - ``task`` / ``robot_type``: plain strings; preprocessor tokenizes task.
    """
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


def _predict_actions(bundle: PolicyBundle, obs: dict, task: str) -> np.ndarray:
    """Serve from queue when possible; infer only on empty queue."""
    now_mono = time.monotonic()
    if _episode_boundary(task, now_mono):
        _reset_episode(bundle)

    batch_size = _serve_batch_size()
    pending = _pop_pending_actions(batch_size)
    if pending:
        _action_queue_state.last_task = task
        _action_queue_state.last_activity_mono = now_mono
        return np.array(pending, dtype=np.float32)

    chunk_np = _run_inference(bundle, obs)
    wire = actions_to_wire_format(chunk_np)
    _action_queue_state.pending.extend(wire)

    served = _pop_pending_actions(batch_size)
    _action_queue_state.last_task = task
    _action_queue_state.last_activity_mono = now_mono
    return np.array(served, dtype=np.float32)


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
        chunk_np = await asyncio.to_thread(
            _predict_actions, _policy_bundle, obs, task
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    _maybe_log_first_predict_debug(state_arr, chunk_np)

    try:
        wire = actions_to_wire_format(chunk_np)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"actions": wire}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=VLA_PORT)
