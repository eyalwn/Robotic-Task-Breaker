from __future__ import annotations

from unittest.mock import MagicMock, patch

from vla.client import VLAClient


def _ok_response():
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"actions": [[0.1, 0.2, 0.3, 0.4]]}
    return resp


def test_predict_passes_state_json_form_field_when_provided():
    client = VLAClient("http://localhost:8765", timeout=5.0)
    with patch("vla.client.requests.post", return_value=_ok_response()) as mock_post:
        client.predict(b"jpeg-bytes", "pick up cube", state=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    _, kwargs = mock_post.call_args
    assert kwargs["data"]["task"] == "pick up cube"
    assert kwargs["data"]["state"] == "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]"


def test_predict_omits_state_form_field_when_none():
    client = VLAClient("http://localhost:8765", timeout=5.0)
    with patch("vla.client.requests.post", return_value=_ok_response()) as mock_post:
        client.predict(b"jpeg-bytes", "pick up cube", state=None)

    _, kwargs = mock_post.call_args
    assert kwargs["data"] == {"task": "pick up cube"}
    assert "image_wrist" not in kwargs["files"]


def test_predict_sends_wrist_multipart_when_provided():
    client = VLAClient("http://localhost:8765", timeout=5.0)
    with patch("vla.client.requests.post", return_value=_ok_response()) as mock_post:
        client.predict(
            b"main-jpeg",
            "pick up cube",
            wrist_image_bytes=b"wrist-jpeg",
        )

    _, kwargs = mock_post.call_args
    assert kwargs["files"]["image"] == ("frame.jpg", b"main-jpeg", "image/jpeg")
    assert kwargs["files"]["image_wrist"] == ("wrist.jpg", b"wrist-jpeg", "image/jpeg")


def test_predict_omits_wrist_multipart_when_none():
    client = VLAClient("http://localhost:8765", timeout=5.0)
    with patch("vla.client.requests.post", return_value=_ok_response()) as mock_post:
        client.predict(b"jpeg-bytes", "pick up cube", wrist_image_bytes=None)

    _, kwargs = mock_post.call_args
    assert "image_wrist" not in kwargs["files"]
