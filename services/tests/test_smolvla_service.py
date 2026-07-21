"""
Tests for services/smolvla_service.py (LeRobot/GPU not required; policy is mocked).
Run with:  .venv/Scripts/python -m pytest services/tests/test_smolvla_service.py -v
"""

from __future__ import annotations

import io
import sys
import time
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


def _post_predict(client, jpeg: bytes, task: str = "pick up the cube", **extra):
    return client.post(
        "/predict",
        files={"image": ("frame.jpg", jpeg, "image/jpeg")},
        data={"task": task, **extra},
    )


@pytest.fixture(autouse=True)
def _reset_smolvla_queue_state():
    svc._clear_action_queue_state()
    yield
    svc._clear_action_queue_state()


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
    bundle.policy.reset.assert_not_called()
    bundle.preprocessor.assert_called_once()
    bundle.policy.predict_action_chunk.assert_called_once()


def test_queue_consumed_over_consecutive_calls(monkeypatch):
    chunk = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    bundle = _mock_bundle(chunk)
    monkeypatch.setenv("SMOLVLA_ACTIONS_PER_PREDICT", "2")

    original = svc._policy_bundle
    svc._policy_bundle = bundle
    try:
        client = TestClient(app, raise_server_exceptions=True)
        jpeg = make_jpeg()
        first = _post_predict(client, jpeg)
        second = _post_predict(client, jpeg)
        third = _post_predict(client, jpeg)
    finally:
        svc._policy_bundle = original

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert first.json()["actions"][0][0] == pytest.approx(0.0)
    assert first.json()["actions"][1][0] == pytest.approx(0.1)
    assert second.json()["actions"][0][0] == pytest.approx(0.2)
    assert second.json()["actions"][1][0] == pytest.approx(0.3)
    assert third.json()["actions"][0][0] == pytest.approx(0.0)
    bundle.policy.reset.assert_not_called()
    assert bundle.policy.predict_action_chunk.call_count == 2
    assert bundle.preprocessor.call_count == 2


def test_task_change_triggers_reset_and_clears_queue(monkeypatch):
    chunk = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    bundle = _mock_bundle(chunk)
    monkeypatch.setenv("SMOLVLA_ACTIONS_PER_PREDICT", "2")

    original = svc._policy_bundle
    svc._policy_bundle = bundle
    try:
        client = TestClient(app, raise_server_exceptions=True)
        jpeg = make_jpeg()
        first = _post_predict(client, jpeg, task="pick up the cube")
        changed = _post_predict(client, jpeg, task="put cube in basket")
    finally:
        svc._policy_bundle = original

    assert first.status_code == 200
    assert changed.status_code == 200
    bundle.policy.reset.assert_called_once()
    assert bundle.policy.predict_action_chunk.call_count == 2
    assert changed.json()["actions"][0][0] == pytest.approx(0.0)


def test_inactivity_timeout_triggers_reset(monkeypatch):
    chunk = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    bundle = _mock_bundle(chunk)
    monkeypatch.setenv("SMOLVLA_ACTIONS_PER_PREDICT", "2")
    monkeypatch.setenv("SMOLVLA_INACTIVITY_TIMEOUT_SEC", "0.01")

    original = svc._policy_bundle
    svc._policy_bundle = bundle
    try:
        client = TestClient(app, raise_server_exceptions=True)
        jpeg = make_jpeg()
        first = _post_predict(client, jpeg)
        time.sleep(0.02)
        second = _post_predict(client, jpeg)
    finally:
        svc._policy_bundle = original

    assert first.status_code == 200
    assert second.status_code == 200
    bundle.policy.reset.assert_called_once()
    assert bundle.policy.predict_action_chunk.call_count == 2


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


def test_prepare_observation_for_inference_tensor_layout():
    import torch

    main = np.full((256, 256, 3), 128, dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)
    obs = svc.build_smolvla_observation(main, "pick cube", state)
    task = obs.pop("task")
    prepared = svc._prepare_observation_for_inference(
        obs, torch.device("cpu"), task=task
    )
    img = prepared["observation.images.image"]
    assert img.shape == (1, 3, 256, 256)
    assert img.dtype == torch.float32
    assert 0.0 <= float(img.min()) <= float(img.max()) <= 1.0
    st = prepared["observation.state"]
    assert st.shape == (1, 8)
    assert st.dtype == torch.float32
    assert prepared["task"] == "pick cube"
    assert prepared["robot_type"] == ""


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


def test_first_predict_debug_logging(monkeypatch, caplog):
    import logging

    chunk = np.array([[0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.9]], dtype=np.float32)
    bundle = _mock_bundle(chunk)

    monkeypatch.setenv("SMOLVLA_DEBUG_FIRST_PREDICT", "1")
    svc._first_predict_debug_logged = False

    original = svc._policy_bundle
    svc._policy_bundle = bundle
    try:
        with caplog.at_level(logging.INFO, logger="smolvla_service"):
            client = TestClient(app, raise_server_exceptions=True)
            jpeg = make_jpeg()
            client.post(
                "/predict",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={"task": "pick up the cube", "state": "[1,2,3,4,5,6,7,8]"},
            )
            client.post(
                "/predict",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={"task": "pick up the cube"},
            )
    finally:
        svc._policy_bundle = original

    debug_msgs = [
        r.message
        for r in caplog.records
        if r.message.startswith("SmolVLA first predict")
    ]
    assert len(debug_msgs) == 2
    assert "state min=" in debug_msgs[0]
    assert "action chunk min=" in debug_msgs[1]
