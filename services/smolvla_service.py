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
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

import config
from utils import configure_logging
configure_logging(fmt="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VLA_PORT = int(os.environ.get("VLA_PORT", "8765"))
VLA_ACTION_CLIP = float(os.environ.get("VLA_ACTION_CLIP", "2.0"))
SMOLVLA_CHECKPOINT = os.environ.get(
    "SMOLVLA_CHECKPOINT", "lerobot/smolvla_vlabench"
).strip()

_VLABENCH_STATE_DIM = 7
_EXPECTED_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)

_policy_bundle: PolicyBundle = False  # type: ignore[assignment]


@dataclass
class PolicyBundle:
    policy: object
    preprocessor: object
    postprocessor: object
    device: object
    state_dim: int
    camera_keys: tuple[str, ...]
    image_size_hw: tuple[int, int]


def decode_image(raw_bytes: bytes, image_size_hw: tuple[int, int]) -> np.ndarray:
    """Decode one JPEG and resize to the checkpoint-required visual shape."""
    target_h, target_w = image_size_hw
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def parse_state_field(state: str, expected_dim: int) -> np.ndarray:
    """Parse and validate the state JSON against the checkpoint contract."""
    if not state:
        raise ValueError(f"state is required ({expected_dim} floats as JSON array)")
    try:
        arr = np.array(_json.loads(state), dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, _json.JSONDecodeError) as exc:
        raise ValueError(f"invalid state JSON: {exc}") from exc
    if arr.size != expected_dim:
        raise ValueError(f"state must have {expected_dim} floats, got {arr.size}")
    return arr


def adapt_state_for_policy(state: np.ndarray, policy_state_dim: int) -> np.ndarray:
    """Adapt Longor's 7-D VLABench state to the checkpoint-required state dim."""
    if policy_state_dim == _VLABENCH_STATE_DIM:
        return state
    if policy_state_dim == 6 and state.size == _VLABENCH_STATE_DIM:
        # smolvla_vlabench currently declares 6-D state; drop gripper component.
        return state[:6]
    raise ValueError(
        f"unsupported state adapter: incoming {state.size}-D, policy expects {policy_state_dim}-D"
    )


def _feature_shape(feature: object, feature_name: str) -> tuple[int, ...]:
    """Extract a numeric shape tuple from one policy input feature."""
    if isinstance(feature, Mapping):
        shape = feature.get("shape")
    else:
        shape = getattr(feature, "shape", None)
    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"{feature_name} is missing a valid shape: {feature!r}")
    try:
        return tuple(int(v) for v in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{feature_name} has non-integer shape: {shape!r}") from exc


def _extract_policy_contract(
    policy: object,
) -> tuple[int, tuple[str, ...], tuple[int, int]]:
    """Return `(state_dim, camera_keys, image_size_hw)` from checkpoint config."""
    cfg = getattr(policy, "config", None)
    if cfg is None:
        raise ValueError("policy is missing config")
    input_features = getattr(cfg, "input_features", None)
    if not isinstance(input_features, Mapping):
        raise ValueError("policy config is missing input_features mapping")

    state_feature = input_features.get("observation.state")
    if state_feature is None:
        raise ValueError("checkpoint is missing observation.state input feature")
    state_shape = _feature_shape(state_feature, "observation.state")
    if len(state_shape) != 1:
        raise ValueError(f"observation.state must be 1-D shape, got {state_shape}")
    state_dim = state_shape[0]

    camera_keys = tuple(
        sorted(key for key in input_features if key.startswith("observation.images."))
    )
    if camera_keys != _EXPECTED_CAMERA_KEYS:
        raise ValueError(
            "unsupported camera contract for checkpoint. "
            f"expected {_EXPECTED_CAMERA_KEYS}, got {camera_keys}"
        )

    ref_shape = _feature_shape(input_features[camera_keys[0]], camera_keys[0])
    if len(ref_shape) != 3:
        raise ValueError(
            f"{camera_keys[0]} must have [C,H,W] shape, got {ref_shape}"
        )
    channels, height, width = ref_shape
    if channels != 3:
        raise ValueError(f"{camera_keys[0]} channels must be 3, got {channels}")
    for key in camera_keys[1:]:
        shape = _feature_shape(input_features[key], key)
        if shape != ref_shape:
            raise ValueError(
                f"all camera shapes must match; {key}={shape}, expected {ref_shape}"
            )
    return state_dim, camera_keys, (height, width)


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
    state_dim, camera_keys, image_size_hw = _extract_policy_contract(policy)
    if state_dim not in {6, _VLABENCH_STATE_DIM}:
        raise ValueError(
            f"checkpoint {SMOLVLA_CHECKPOINT!r} expects unsupported {state_dim}-D state; "
            f"Longor supports only 6-D or {_VLABENCH_STATE_DIM}-D policy state."
        )
    if state_dim == 6:
        log.warning(
            "checkpoint %s expects 6-D state; applying explicit 7->6 adapter (drop gripper)",
            SMOLVLA_CHECKPOINT,
        )
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
        state_dim=state_dim,
        camera_keys=camera_keys,
        image_size_hw=image_size_hw,
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

    expected_images = len(_policy_bundle.camera_keys)
    if len(images) != expected_images:
        raise HTTPException(
            status_code=400,
            detail=f"expected {expected_images} images, got {len(images)}",
        )

    try:
        decoded = []
        for upload in images:
            raw = await upload.read()
            decoded.append(decode_image(raw, _policy_bundle.image_size_hw))
        state_arr = parse_state_field(state, _VLABENCH_STATE_DIM)
        state_arr = adapt_state_for_policy(state_arr, _policy_bundle.state_dim)
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
