"""Hermetic tests for SmolVLA service wire-format helpers and /predict."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import config
import smolvla_service as svc
from smolvla_service import (
    actions_to_wire_format,
    app,
    build_smolvla_observation,
    truncate_action_chunk,
)

TestClient = pytest.importorskip("starlette.testclient").TestClient


def valid_state_json() -> str:
    return json.dumps([0.0] * 7)


def make_jpeg(
    width: int = 64, height: int = 64, color: tuple[int, int, int] = (120, 80, 40)
) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_bundle() -> object:
    return SimpleNamespace(
        camera_keys=(
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ),
        image_size_hw=(256, 256),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        device=object(),
    )


def _mock_policy_contract(
    *,
    state_dim: int = 7,
    camera_keys: tuple[str, ...] = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    ),
    camera_shape: tuple[int, int, int] = (3, 256, 256),
) -> object:
    input_features = {"observation.state": {"shape": (state_dim,)}}
    for key in camera_keys:
        input_features[key] = {"shape": camera_shape}
    return SimpleNamespace(config=SimpleNamespace(input_features=input_features))


def test_truncates_to_max_actions() -> None:
    chunk = np.arange(30, dtype=np.float32).reshape(10, 3)
    truncated = truncate_action_chunk(chunk, 3)
    assert truncated.shape == (3, 3)
    assert truncated[0, 0] == 0.0
    assert truncated[2, 0] == 6.0


def test_wire_format_truncated_to_config_limit() -> None:
    chunk = np.full((10, 7), 3.0, dtype=np.float32)
    truncated = truncate_action_chunk(chunk, config.VLA_ACTIONS_PER_CHUNK)
    wire = actions_to_wire_format(truncated)
    assert len(wire) == config.VLA_ACTIONS_PER_CHUNK
    assert all(len(action) == 7 for action in wire)
    assert wire[0][0] == config.ACTION_CLIP


def test_build_smolvla_observation_uses_three_cameras() -> None:
    front = np.full((256, 256, 3), 10, dtype=np.uint8)
    second = np.full((256, 256, 3), 20, dtype=np.uint8)
    wrist = np.full((256, 256, 3), 30, dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)

    obs = build_smolvla_observation(front, second, wrist, "pick fruit", state)

    np.testing.assert_array_equal(obs["observation.images.camera1"], front)
    np.testing.assert_array_equal(obs["observation.images.camera2"], second)
    np.testing.assert_array_equal(obs["observation.images.camera3"], wrist)
    np.testing.assert_array_equal(obs["observation.state"], state)
    assert obs["task"] == "pick fruit"


def test_extract_policy_contract_uses_visual_checkpoint_layout() -> None:
    policy = _mock_policy_contract(state_dim=6, camera_shape=(3, 224, 320))
    camera_keys, image_size_hw = svc._extract_policy_contract(policy)
    assert camera_keys == (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    assert image_size_hw == (224, 320)


def test_extract_policy_contract_rejects_unexpected_camera_keys() -> None:
    policy = _mock_policy_contract(
        camera_keys=("observation.images.image", "observation.images.wrist_image")
    )
    with pytest.raises(ValueError, match="unsupported camera contract"):
        svc._extract_policy_contract(policy)


def test_health() -> None:
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "smolvla"}


def test_predict_with_mock_policy() -> None:
    bundle = _mock_bundle()
    original = svc._policy_bundle
    svc._policy_bundle = bundle
    try:
        with patch.object(svc, "_run_inference", return_value=np.zeros((10, 7), dtype=np.float32)):
            client = TestClient(app, raise_server_exceptions=True)
            jpeg = make_jpeg()
            resp = client.post(
                "/predict",
                files=[
                    ("images", ("cam0.jpg", jpeg, "image/jpeg")),
                    ("images", ("cam1.jpg", jpeg, "image/jpeg")),
                    ("images", ("cam2.jpg", jpeg, "image/jpeg")),
                ],
                data={"task": "pick up the cube", "state": valid_state_json()},
            )
    finally:
        svc._policy_bundle = original

    assert resp.status_code == 200, resp.text
    body = resp.json()
    actions = body["actions"]
    assert len(actions) == config.VLA_ACTIONS_PER_CHUNK
    for row in actions:
        assert isinstance(row, list) and len(row) == 7


def _post_predict(client: TestClient, *, images: int = 3, state: str = valid_state_json()) -> object:
    jpeg = make_jpeg()
    files = [("images", (f"cam{i}.jpg", jpeg, "image/jpeg")) for i in range(images)]
    return client.post(
        "/predict",
        files=files,
        data={"task": "pick up the cube", "state": state},
    )


def test_predict_rejects_two_images() -> None:
    original = svc._policy_bundle
    svc._policy_bundle = _mock_bundle()
    try:
        client = TestClient(app, raise_server_exceptions=True)
        resp = _post_predict(client, images=2)
    finally:
        svc._policy_bundle = original

    assert resp.status_code == 400
    assert "expected 3 images" in resp.json()["detail"]


def test_predict_rejects_wrong_state_length() -> None:
    original = svc._policy_bundle
    svc._policy_bundle = _mock_bundle()
    try:
        client = TestClient(app, raise_server_exceptions=True)
        resp = _post_predict(client, state=json.dumps([0.0, 0.0, 0.0]))
    finally:
        svc._policy_bundle = original

    assert resp.status_code == 400
    assert "7 floats" in resp.json()["detail"]


def test_predict_rejects_invalid_state_json() -> None:
    original = svc._policy_bundle
    svc._policy_bundle = _mock_bundle()
    try:
        client = TestClient(app, raise_server_exceptions=True)
        resp = _post_predict(client, state="not-json")
    finally:
        svc._policy_bundle = original

    assert resp.status_code == 400
    assert "invalid state JSON" in resp.json()["detail"]
