"""VLA HTTP client wire-contract tests (no network)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from vla_client import VLAClient, VLAError


def _ok_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_health_get_request() -> None:
    client = VLAClient("http://127.0.0.1:8765", timeout=5.0)
    with patch("vla_client.requests.get", return_value=_ok_response({"ok": True})) as mock_get:
        result = client.health()
    mock_get.assert_called_once_with("http://127.0.0.1:8765/health", timeout=5.0)
    assert result == {"ok": True}


def test_predict_multipart_request_shape() -> None:
    client = VLAClient("http://127.0.0.1:8765", timeout=5.0)
    actions = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]]
    images = (b"main-jpeg", b"wrist-jpeg")
    with patch(
        "vla_client.requests.post",
        return_value=_ok_response({"actions": actions}),
    ) as mock_post:
        result = client.predict(images, [1.0, 2.0, 3.0], "pick up cube")

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://127.0.0.1:8765/predict"
    assert kwargs["timeout"] == 5.0
    assert kwargs["files"] == [
        ("images", ("cam0.jpg", b"main-jpeg", "image/jpeg")),
        ("images", ("cam1.jpg", b"wrist-jpeg", "image/jpeg")),
    ]
    assert kwargs["data"]["task"] == "pick up cube"
    assert kwargs["data"]["state"] == json.dumps([1.0, 2.0, 3.0])
    assert result == actions


def test_predict_rejects_empty_images() -> None:
    client = VLAClient("http://127.0.0.1:8765", timeout=5.0)
    with pytest.raises(VLAError, match="at least one image"):
        client.predict((), [], "task")


def test_predict_clips_actions() -> None:
    client = VLAClient("http://127.0.0.1:8765", timeout=5.0)
    raw = [[99.0, -99.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    with patch(
        "vla_client.requests.post",
        return_value=_ok_response({"actions": raw}),
    ):
        clipped = client.predict((b"i", b"w"), [], "task")
    assert clipped[0][0] == 2.0
    assert clipped[0][1] == -2.0


def test_predict_rejects_empty_actions() -> None:
    client = VLAClient("http://127.0.0.1:8765", timeout=5.0)
    with patch(
        "vla_client.requests.post",
        return_value=_ok_response({"actions": []}),
    ):
        with pytest.raises(VLAError, match="non-empty 'actions' list"):
            client.predict((b"i", b"w"), [], "task")


def test_predict_rejects_bad_action_dim() -> None:
    client = VLAClient("http://127.0.0.1:8765", timeout=5.0)
    with patch(
        "vla_client.requests.post",
        return_value=_ok_response({"actions": [[1.0, 2.0]]}),
    ):
        with pytest.raises(VLAError, match="must be 7 floats"):
            client.predict((b"i", b"w"), [], "task")


def test_predict_rejects_non_finite() -> None:
    client = VLAClient("http://127.0.0.1:8765", timeout=5.0)
    with patch(
        "vla_client.requests.post",
        return_value=_ok_response({"actions": [[float("nan")] * 7]}),
    ):
        with pytest.raises(VLAError, match="not finite"):
            client.predict((b"i", b"w"), [], "task")


def test_health_failure_wraps_vla_error() -> None:
    client = VLAClient("http://127.0.0.1:8765", timeout=5.0)
    with patch(
        "vla_client.requests.get",
        side_effect=requests.ConnectionError("refused"),
    ):
        with pytest.raises(VLAError, match="failed"):
            client.health()
