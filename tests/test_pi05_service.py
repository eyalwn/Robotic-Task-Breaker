"""Hermetic tests for Pi0.5 service wire-format helpers."""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

import config
import pi05_service as svc
from pi05_service import (
    actions_to_wire_format,
    app,
    build_openpi_observation,
    truncate_action_chunk,
)

TestClient = pytest.importorskip("starlette.testclient").TestClient


def valid_state_json() -> str:
    return json.dumps([0.0] * 8)


def make_jpeg(
    width: int = 64, height: int = 64, color: tuple[int, int, int] = (120, 80, 40)
) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_truncates_to_max_actions() -> None:
    chunk = np.arange(30, dtype=np.float32).reshape(10, 3)
    truncated = truncate_action_chunk(chunk, 3)
    assert truncated.shape == (3, 3)
    assert truncated[0, 0] == 0.0
    assert truncated[2, 0] == 6.0


def test_truncates_rejects_non_positive_max() -> None:
    chunk = np.ones((2, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="max_actions must be positive"):
        truncate_action_chunk(chunk, 0)


def test_wire_format_truncated_to_config_limit() -> None:
    chunk = np.full((10, 7), 3.0, dtype=np.float32)
    truncated = truncate_action_chunk(chunk, config.VLA_ACTIONS_PER_CHUNK)
    wire = actions_to_wire_format(truncated)
    assert len(wire) == config.VLA_ACTIONS_PER_CHUNK
    assert all(len(action) == 7 for action in wire)
    clip = float(svc.VLA_ACTION_CLIP)
    assert wire[0][0] == clip


def test_actions_to_wire_format_returns_all_seven_dims() -> None:
    chunk = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.9],
            [2.5, -1.0, 0.0, 0.1, 0.2, 0.3, 1.0],
        ],
        dtype=np.float32,
    )
    wire = actions_to_wire_format(chunk)
    assert len(wire) == 2
    assert wire[0] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.9])
    clip = float(svc.VLA_ACTION_CLIP)
    assert wire[1] == pytest.approx([clip, -1.0, 0.0, 0.1, 0.2, 0.3, 1.0])


def test_build_openpi_observation_uses_wrist_when_provided() -> None:
    main = np.full((224, 224, 3), 10, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 20, dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)

    obs = build_openpi_observation("pi05_libero", main, "pick cube", state, wrist)

    np.testing.assert_array_equal(obs["observation/image"], main)
    np.testing.assert_array_equal(obs["observation/wrist_image"], wrist)
    assert obs["prompt"] == "pick cube"
    np.testing.assert_array_equal(obs["observation/state"], state)


def test_health() -> None:
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "pi05"}


def test_predict_with_mock_policy() -> None:
    policy = MagicMock()
    policy.infer.return_value = {"actions": np.zeros((10, 8), dtype=np.float32)}

    original = svc._policy
    svc._policy = policy
    try:
        client = TestClient(app, raise_server_exceptions=True)
        jpeg = make_jpeg()
        resp = client.post(
            "/predict",
            files=[
                ("images", ("cam0.jpg", jpeg, "image/jpeg")),
                ("images", ("cam1.jpg", jpeg, "image/jpeg")),
            ],
            data={"task": "pick up the cube", "state": valid_state_json()},
        )
    finally:
        svc._policy = original

    assert resp.status_code == 200, resp.text
    body = resp.json()
    actions = body["actions"]
    assert len(actions) == config.VLA_ACTIONS_PER_CHUNK
    for row in actions:
        assert isinstance(row, list) and len(row) == 7
    policy.infer.assert_called_once()


def test_predict_with_wrist_image(monkeypatch: pytest.MonkeyPatch) -> None:
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
            files=[
                ("images", ("cam0.jpg", main_jpeg, "image/jpeg")),
                ("images", ("cam1.jpg", wrist_jpeg, "image/jpeg")),
            ],
            data={"task": "pick up the cube", "state": valid_state_json()},
        )
    finally:
        svc._policy = original

    assert resp.status_code == 200, resp.text
    obs = policy.infer.call_args[0][0]
    assert not np.array_equal(
        obs["observation/wrist_image"],
        np.zeros_like(obs["observation/image"]),
    )


def _post_predict(client: TestClient, *, images: int = 2, state: str = valid_state_json()) -> object:
    jpeg = make_jpeg()
    files = [("images", (f"cam{i}.jpg", jpeg, "image/jpeg")) for i in range(images)]
    return client.post(
        "/predict",
        files=files,
        data={"task": "pick up the cube", "state": state},
    )


def test_predict_rejects_single_image() -> None:
    original = svc._policy
    svc._policy = MagicMock()
    try:
        client = TestClient(app, raise_server_exceptions=True)
        resp = _post_predict(client, images=1)
    finally:
        svc._policy = original

    assert resp.status_code == 400
    assert "expected 2 images" in resp.json()["detail"]


def test_predict_rejects_wrong_state_length() -> None:
    original = svc._policy
    svc._policy = MagicMock()
    try:
        client = TestClient(app, raise_server_exceptions=True)
        resp = _post_predict(client, state=json.dumps([0.0, 0.0, 0.0]))
    finally:
        svc._policy = original

    assert resp.status_code == 400
    assert "8 floats" in resp.json()["detail"]


def test_predict_rejects_invalid_state_json() -> None:
    original = svc._policy
    svc._policy = MagicMock()
    try:
        client = TestClient(app, raise_server_exceptions=True)
        resp = _post_predict(client, state="not-json")
    finally:
        svc._policy = original

    assert resp.status_code == 400
    assert "invalid state JSON" in resp.json()["detail"]
