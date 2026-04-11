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


def make_jpeg(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 80, 40))
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
        assert isinstance(row, list) and len(row) == 4
        assert all(isinstance(v, float) for v in row)
    policy.infer.assert_called_once()


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
