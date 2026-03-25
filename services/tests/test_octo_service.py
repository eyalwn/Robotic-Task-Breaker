"""
Tests for services/octo_service.py.
Run with:  .octo-venv/Scripts/python -m pytest services/tests/ -v
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image
TestClient = pytest.importorskip("starlette.testclient").TestClient

# Make `services/` importable when running from repo root / longor/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import services.octo_service as svc
from services.octo_service import IMAGE_SIZE, WINDOW, WRIST_SIZE, app, build_obs, decode_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VLA_MODEL_DIR = os.environ.get("VLA_MODEL_DIR", str(Path.home() / ".cache" / "octo-small-1.5"))
MODEL_AVAILABLE = Path(VLA_MODEL_DIR).exists()
skip_if_no_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason=f"VLA_MODEL_DIR not found: {VLA_MODEL_DIR}",
)


def make_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Generate a small solid-colour JPEG in memory."""
    img = Image.new("RGB", (width, height), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. test_health — no model needed, no lifespan
# ---------------------------------------------------------------------------

def test_health():
    # Don't enter the lifespan context — health has no model dependency.
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "octo-small-1.5"}


# ---------------------------------------------------------------------------
# 2. test_predict_image_preprocessing — unit-test decode_image in isolation
# ---------------------------------------------------------------------------

def test_predict_image_preprocessing():
    jpeg = make_jpeg(32, 32)
    img = decode_image(jpeg)
    assert img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"unexpected shape {img.shape}"
    assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# 3. test_predict_obs_dict_shape — obs dict shapes / dtypes
# ---------------------------------------------------------------------------

def test_predict_obs_dict_shape():
    img_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    obs = build_obs(img_np)

    # image_primary
    assert obs["image_primary"].shape == (1, WINDOW, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert obs["image_primary"].dtype == np.uint8

    # image_wrist
    assert obs["image_wrist"].shape == (1, WINDOW, WRIST_SIZE, WRIST_SIZE, 3)
    assert obs["image_wrist"].dtype == np.uint8

    # timestep_pad_mask — critical key that was missing before the fix
    assert obs["timestep_pad_mask"].shape == (1, WINDOW)
    assert obs["timestep_pad_mask"].dtype == bool

    # pad_mask_dict sub-keys
    pmd = obs["pad_mask_dict"]
    assert pmd["image_primary"].shape == (1, WINDOW)
    assert pmd["image_wrist"].shape == (1, WINDOW)
    assert pmd["timestep"].shape == (1, WINDOW)

    # task_completed / timestep scalars
    assert obs["task_completed"].shape == (1, WINDOW, 4)
    assert obs["timestep"].shape == (1, WINDOW)
    assert obs["timestep"].dtype == np.int32


# ---------------------------------------------------------------------------
# 4. test_predict_with_real_model — load model, call sample_actions directly
# ---------------------------------------------------------------------------

@skip_if_no_model
def test_predict_with_real_model():
    import jax
    from octo.model.octo_model import OctoModel

    model = OctoModel.load_pretrained(VLA_MODEL_DIR)

    img_np = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    obs = build_obs(img_np)
    task = model.create_tasks(texts=["pick up the cube"])

    rng = jax.random.PRNGKey(42)
    actions = model.sample_actions(obs, task, rng=rng)

    # shape: (batch=1, action_horizon, action_dim)
    assert actions.ndim == 3, f"expected 3-d actions, got shape {actions.shape}"
    assert actions.shape[0] == 1
    assert actions.shape[2] >= 4, f"need at least 4 action dims, got {actions.shape[2]}"

    action_vec = np.array(actions[0, 0, :4]).tolist()
    assert len(action_vec) == 4
    assert all(isinstance(v, float) for v in action_vec)


# ---------------------------------------------------------------------------
# 5. test_predict_endpoint_with_real_model — full TestClient POST /predict
# ---------------------------------------------------------------------------

@skip_if_no_model
def test_predict_endpoint_with_real_model():
    with patch("services.octo_service.Path") as MockPath:
        # Make the path-existence check pass
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.__str__ = lambda self: VLA_MODEL_DIR
        MockPath.return_value = mock_path_instance
        MockPath.home.return_value = Path.home()

        with TestClient(app) as client:
            jpeg = make_jpeg()
            resp = client.post(
                "/predict",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={"task": "pick up the cube"},
            )

    assert resp.status_code == 200, f"got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "action" in body
    action = body["action"]
    assert isinstance(action, list)
    assert len(action) == 4
    assert all(isinstance(v, float) for v in action)


# ---------------------------------------------------------------------------
# 6. test_predict_bad_image — garbage bytes → 400
# ---------------------------------------------------------------------------

def test_predict_bad_image():
    # Inject a mock model so the endpoint doesn't 503 before reaching image decode
    original = svc._model
    svc._model = MagicMock()
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/predict",
            files={"image": ("bad.jpg", b"not-a-jpeg-at-all", "image/jpeg")},
        )
        assert resp.status_code == 400
    finally:
        svc._model = original
