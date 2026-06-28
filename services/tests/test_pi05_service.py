"""
Tests for services/pi05_service.py (OpenPI not required; policy is mocked).
Run with:  .venv/Scripts/python -m pytest services/tests/test_pi05_service.py -v
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

TestClient = pytest.importorskip("starlette.testclient").TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import services.pi05_service as svc
from services.pi05_service import app


def make_jpeg(width: int = 64, height: int = 64, color: tuple[int, int, int] = (120, 80, 40)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "pi05"}


def test_predict_with_mock_policy():
    policy = MagicMock()
    policy.infer.return_value = {"actions": np.zeros((3, 8), dtype=np.float32)}

    original = svc._policy
    svc._policy = policy
    try:
        client = TestClient(app, raise_server_exceptions=True)
        jpeg = make_jpeg()
        resp = client.post(
            "/predict",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"task": "pick up the cube"},
        )
    finally:
        svc._policy = original

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "actions" in body
    actions = body["actions"]
    assert len(actions) == 3
    for row in actions:
        assert isinstance(row, list) and len(row) == 7
        assert all(isinstance(v, float) for v in row)
    policy.infer.assert_called_once()


def test_actions_to_wire_format_returns_all_seven_dims():
    chunk = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.9],
            [2.5, -1.0, 0.0, 0.1, 0.2, 0.3, 1.0],
        ],
        dtype=np.float32,
    )
    wire = svc.actions_to_wire_format(chunk)
    assert len(wire) == 2
    assert wire[0] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.9])
    clip = float(svc.VLA_ACTION_CLIP)
    assert wire[1] == pytest.approx([clip, -1.0, 0.0, 0.1, 0.2, 0.3, 1.0])


def test_build_openpi_observation_uses_wrist_when_provided():
    main = np.full((224, 224, 3), 10, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 20, dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)

    obs = svc.build_openpi_observation("pi05_libero", main, "pick cube", state, wrist)

    np.testing.assert_array_equal(obs["observation/image"], main)
    np.testing.assert_array_equal(obs["observation/wrist_image"], wrist)
    assert obs["prompt"] == "pick cube"
    np.testing.assert_array_equal(obs["observation/state"], state)


def test_build_openpi_observation_zeros_wrist_when_missing():
    main = np.full((224, 224, 3), 10, dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)

    obs = svc.build_openpi_observation("pi05_libero", main, "pick cube", state)

    np.testing.assert_array_equal(obs["observation/wrist_image"], np.zeros_like(main))


def test_predict_with_wrist_image(monkeypatch):
    monkeypatch.setattr(svc, "OPENPI_CONFIG", "pi05_libero")
    policy = MagicMock()
    policy.infer.return_value = {"actions": np.zeros((1, 8), dtype=np.float32)}

    original = svc._policy
    svc._policy = policy
    try:
        client = TestClient(app, raise_server_exceptions=True)
        main_jpeg = make_jpeg(color=(120, 80, 40))
        wrist_jpeg = make_jpeg(color=(40, 80, 120))
        resp = client.post(
            "/predict",
            files={
                "image": ("frame.jpg", main_jpeg, "image/jpeg"),
                "image_wrist": ("wrist.jpg", wrist_jpeg, "image/jpeg"),
            },
            data={"task": "pick up the cube"},
        )
    finally:
        svc._policy = original

    assert resp.status_code == 200, resp.text
    policy.infer.assert_called_once()
    obs = policy.infer.call_args[0][0]
    assert "observation/wrist_image" in obs
    assert not np.array_equal(
        obs["observation/wrist_image"],
        np.zeros_like(obs["observation/image"]),
    )


def test_predict_model_not_loaded():
    original = svc._policy
    svc._policy = None
    try:
        client = TestClient(app, raise_server_exceptions=False)
        jpeg = make_jpeg()
        resp = client.post(
            "/predict",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"task": "pick up the cube"},
        )
    finally:
        svc._policy = original

    assert resp.status_code == 503


def test_predict_bad_image():
    policy = MagicMock()
    original = svc._policy
    svc._policy = policy
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/predict",
            files={"image": ("bad.jpg", b"not-a-jpeg-at-all", "image/jpeg")},
        )
    finally:
        svc._policy = original

    assert resp.status_code == 400
