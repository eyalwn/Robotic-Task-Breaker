"""
Tests for services/smolvla_service.py (LeRobot/GPU not required; policy is mocked).
Run with:  .venv/Scripts/python -m pytest services/tests/test_smolvla_service.py -v
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

import services.smolvla_service as svc
from services.smolvla_service import PolicyBundle, app


def make_jpeg(width: int = 64, height: int = 64, color: tuple[int, int, int] = (120, 80, 40)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_bundle(chunk_np: np.ndarray) -> PolicyBundle:
    import torch

    policy = MagicMock()
    policy.predict_action_chunk.return_value = torch.from_numpy(chunk_np).unsqueeze(0)
    preprocessor = MagicMock(side_effect=lambda obs: obs)
    postprocessor = MagicMock(side_effect=lambda row: row)
    return PolicyBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=torch.device("cpu"),
    )


def test_health():
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "smolvla"}


def test_predict_with_mock_policy():
    chunk = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.9],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    bundle = _mock_bundle(chunk)

    original = svc._policy_bundle
    svc._policy_bundle = bundle
    try:
        client = TestClient(app, raise_server_exceptions=True)
        jpeg = make_jpeg()
        resp = client.post(
            "/predict",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"task": "pick up the cube"},
        )
    finally:
        svc._policy_bundle = original

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "actions" in body
    actions = body["actions"]
    assert len(actions) == 2
    for row in actions:
        assert isinstance(row, list) and len(row) == 7
        assert all(isinstance(v, float) for v in row)
    bundle.policy.reset.assert_called_once()
    bundle.preprocessor.assert_called_once()
    bundle.policy.predict_action_chunk.assert_called_once()


def test_predict_with_wrist_image():
    chunk = np.zeros((1, 7), dtype=np.float32)
    bundle = _mock_bundle(chunk)

    original = svc._policy_bundle
    svc._policy_bundle = bundle
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
        svc._policy_bundle = original

    assert resp.status_code == 200, resp.text
    obs = bundle.preprocessor.call_args[0][0]
    assert "observation.images.image2" in obs
    assert not np.array_equal(
        obs["observation.images.image2"],
        np.zeros_like(obs["observation.images.image"]),
    )


def test_actions_to_wire_format_clips_values():
    chunk = np.array([[2.5, -3.0, 0.0, 0.1, 0.2, 0.3, 1.0]], dtype=np.float32)
    wire = svc.actions_to_wire_format(chunk)
    clip = float(svc.VLA_ACTION_CLIP)
    assert wire[0] == pytest.approx([clip, -clip, 0.0, 0.1, 0.2, 0.3, 1.0])


def test_actions_to_wire_format_rejects_bad_shape():
    with pytest.raises(ValueError, match="2D"):
        svc.actions_to_wire_format(np.zeros((7,), dtype=np.float32))
    with pytest.raises(ValueError, match=">= 7"):
        svc.actions_to_wire_format(np.zeros((2, 5), dtype=np.float32))


def test_parse_state_field_malformed_uses_zeros():
    zeros = svc.parse_state_field("not-json")
    np.testing.assert_array_equal(zeros, np.zeros(8, dtype=np.float32))
    zeros = svc.parse_state_field("[1, 2, 3]")
    np.testing.assert_array_equal(zeros, np.zeros(8, dtype=np.float32))


def test_build_smolvla_observation_zeros_wrist_when_missing():
    main = np.full((256, 256, 3), 10, dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)
    obs = svc.build_smolvla_observation(main, "pick cube", state)
    np.testing.assert_array_equal(obs["observation.images.image"], main)
    np.testing.assert_array_equal(obs["observation.images.image2"], np.zeros_like(main))
    assert obs["task"] == "pick cube"
    np.testing.assert_array_equal(obs["observation.state"], state)


def test_predict_model_not_loaded():
    original = svc._policy_bundle
    svc._policy_bundle = None
    try:
        client = TestClient(app, raise_server_exceptions=False)
        jpeg = make_jpeg()
        resp = client.post(
            "/predict",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"task": "pick up the cube"},
        )
    finally:
        svc._policy_bundle = original

    assert resp.status_code == 503


def test_predict_bad_image():
    bundle = _mock_bundle(np.zeros((1, 7), dtype=np.float32))
    original = svc._policy_bundle
    svc._policy_bundle = bundle
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/predict",
            files={"image": ("bad.jpg", b"not-a-jpeg-at-all", "image/jpeg")},
        )
    finally:
        svc._policy_bundle = original

    assert resp.status_code == 400


def test_predict_invalid_action_shape_returns_500():
    import torch

    policy = MagicMock()
    policy.predict_action_chunk.return_value = torch.zeros(1, 2, 5)
    bundle = PolicyBundle(
        policy=policy,
        preprocessor=MagicMock(side_effect=lambda obs: obs),
        postprocessor=MagicMock(side_effect=lambda row: row),
        device=torch.device("cpu"),
    )

    original = svc._policy_bundle
    svc._policy_bundle = bundle
    try:
        client = TestClient(app, raise_server_exceptions=False)
        jpeg = make_jpeg()
        resp = client.post(
            "/predict",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"task": "pick up the cube"},
        )
    finally:
        svc._policy_bundle = original

    assert resp.status_code == 500
    assert "7" in resp.json()["detail"]
