"""Hermetic tests for pi0fast service wire-format helpers and /predict."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import config
import pi0fast_service as svc
from pi0fast_service import (
    PI0FAST_CHECKPOINT_REVISION,
    _FAST_TOKENIZER_REPO,
    _FAST_TOKENIZER_REVISION,
    _download_checkpoint,
    _download_fast_tokenizer,
    _with_pinned_fast_tokenizer,
    actions_to_wire_format,
    app,
    build_pi0fast_observation,
    resolve_checkpoint_repo,
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


def test_resolve_checkpoint_repo_maps_srd_name() -> None:
    resolved = resolve_checkpoint_repo("VLABench/pi0-fast-primitive-10task-deltachunk")
    assert resolved == "VLABench/pi0-fast-ft-primitive-10task-deltachunk"


def test_download_checkpoint_uses_inference_only_patterns() -> None:
    pytest.importorskip("huggingface_hub")
    calls = {}

    def _fake_snapshot_download(
        repo_id,
        revision,
        allow_patterns,
        ignore_patterns,
    ):
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        calls["allow_patterns"] = allow_patterns
        calls["ignore_patterns"] = ignore_patterns
        return "/tmp/ckpt"

    with patch("huggingface_hub.snapshot_download", _fake_snapshot_download):
        out = _download_checkpoint("foo/bar")
    assert out == "/tmp/ckpt"
    assert calls["repo_id"] == "foo/bar"
    assert calls["revision"] == PI0FAST_CHECKPOINT_REVISION
    assert "params/**" in calls["allow_patterns"]
    assert "assets/**" in calls["allow_patterns"]
    assert calls["ignore_patterns"] == ["train_state/**"]


def test_download_fast_tokenizer_uses_pinned_revision() -> None:
    pytest.importorskip("huggingface_hub")
    calls = {}

    def _fake_snapshot_download(repo_id, revision):
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        return "/tmp/fast"

    with patch("huggingface_hub.snapshot_download", _fake_snapshot_download):
        out = _download_fast_tokenizer()
    assert out == "/tmp/fast"
    assert calls == {
        "repo_id": _FAST_TOKENIZER_REPO,
        "revision": _FAST_TOKENIZER_REVISION,
    }


@dataclass(frozen=True)
class _ModelConfig:
    fast_model_tokenizer_kwargs: dict[str, str] | None = None


@dataclass(frozen=True)
class _TrainConfig:
    model: _ModelConfig


def test_pinned_fast_tokenizer_replaces_model_path_only() -> None:
    train_cfg = _TrainConfig(
        model=_ModelConfig(fast_model_tokenizer_kwargs={"other_option": "value"})
    )
    updated = _with_pinned_fast_tokenizer(train_cfg, "/tmp/fast")
    assert updated.model.fast_model_tokenizer_kwargs == {
        "other_option": "value",
        "fast_tokenizer_path": "/tmp/fast",
    }
    assert train_cfg.model.fast_model_tokenizer_kwargs == {"other_option": "value"}


def test_truncates_to_max_actions() -> None:
    chunk = np.arange(30, dtype=np.float32).reshape(10, 3)
    truncated = truncate_action_chunk(chunk, 3)
    assert truncated.shape == (3, 3)


def test_wire_format_preserves_raw_vectors_including_pi_roll() -> None:
    chunk = np.array(
        [
            [0.1, 0.2, 0.3, 3.14159265, -1.5, 0.0, 1.0],
            [2.5, -1.0, 0.0, 0.1, 0.2, 0.3, -1.0],
        ],
        dtype=np.float32,
    )
    wire = actions_to_wire_format(chunk)
    assert wire[0] == pytest.approx([0.1, 0.2, 0.3, 3.14159265, -1.5, 0.0, 1.0])
    assert wire[1] == pytest.approx([2.5, -1.0, 0.0, 0.1, 0.2, 0.3, -1.0])


def test_build_pi0fast_observation_matches_openpi_rgb_index_order() -> None:
    """OpenPI VLABench eval assigns rgb[0,1,2] -> image, second_image, wrist_image."""
    frames = [
        np.full((224, 224, 3), 10, dtype=np.uint8),
        np.full((224, 224, 3), 20, dtype=np.uint8),
        np.full((224, 224, 3), 30, dtype=np.uint8),
    ]
    state = np.arange(7, dtype=np.float32)
    obs = build_pi0fast_observation(*frames, "pick fruit", state)
    for key, frame in zip(svc.PI0FAST_OBSERVATION_IMAGE_KEYS, frames, strict=True):
        np.testing.assert_array_equal(obs[key], frame)


def test_build_pi0fast_observation_uses_three_cameras() -> None:
    front = np.full((224, 224, 3), 10, dtype=np.uint8)
    second = np.full((224, 224, 3), 20, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 30, dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)

    obs = build_pi0fast_observation(front, second, wrist, "pick fruit", state)

    np.testing.assert_array_equal(obs["observation/image"], front)
    np.testing.assert_array_equal(obs["observation/second_image"], second)
    np.testing.assert_array_equal(obs["observation/wrist_image"], wrist)
    np.testing.assert_array_equal(obs["observation/state"], state)
    assert obs["prompt"] == "pick fruit"


def test_health() -> None:
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "pi0fast"}


def test_predict_with_mock_policy() -> None:
    policy = MagicMock()
    policy.infer.return_value = {"actions": np.zeros((10, 7), dtype=np.float32)}

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
                ("images", ("cam2.jpg", jpeg, "image/jpeg")),
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
    obs = policy.infer.call_args[0][0]
    assert set(obs) >= {
        "observation/image",
        "observation/second_image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }


def test_predict_rejects_two_images() -> None:
    original = svc._policy
    svc._policy = MagicMock()
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

    assert resp.status_code == 400
    assert "expected 3 images" in resp.json()["detail"]


def test_predict_rejects_wrong_state_length() -> None:
    original = svc._policy
    svc._policy = MagicMock()
    try:
        client = TestClient(app, raise_server_exceptions=True)
        jpeg = make_jpeg()
        resp = client.post(
            "/predict",
            files=[("images", (f"cam{i}.jpg", jpeg, "image/jpeg")) for i in range(3)],
            data={"task": "pick up the cube", "state": json.dumps([0.0, 0.0, 0.0])},
        )
    finally:
        svc._policy = original

    assert resp.status_code == 400
    assert "7 floats" in resp.json()["detail"]

