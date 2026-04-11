"""Integration tests for the VLA HTTP client ↔ server round-trip.

Run with:  pytest -m integration
These tests require the VLA HTTP service to be started separately before running.
"""
from __future__ import annotations

import math
import os

import pytest
import requests

from sim.maniskill_backend import ManiSkillArm
from vla import VLAConfig, create_vla_manipulation

from conftest import FakeEnv

pytestmark = pytest.mark.integration

_DEFAULT_URL = "http://localhost:8765"


def _vla_url() -> str:
    port = os.environ.get("VLA_PORT")
    if port:
        return f"http://localhost:{port}"
    return _DEFAULT_URL


def _synthetic_jpeg(width: int = 256, height: int = 256) -> bytes:
    """Return a solid-colour 256×256 JPEG as raw bytes."""
    try:
        from PIL import Image
        import io

        img = Image.new("RGB", (width, height), color=(100, 149, 237))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        import cv2
        import numpy as np

        img = np.full((height, width, 3), (237, 149, 100), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()


@pytest.fixture(scope="module")
def live_url():
    """Resolve the VLA service URL and skip the test if the server is unreachable."""
    url = _vla_url()
    try:
        requests.get(f"{url}/health", timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip(f"VLA service not reachable at {url}")
    return url


def test_health_check(live_url):
    resp = requests.get(f"{live_url}/health", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"


def test_single_predict_roundtrip(live_url):
    jpeg = _synthetic_jpeg()
    resp = requests.post(
        f"{live_url}/predict",
        files={"image": ("frame.jpg", jpeg, "image/jpeg")},
        data={"task": "pick the cube"},
        timeout=60,
    )
    assert resp.status_code == 200

    body = resp.json()
    # Server may return a single action or a chunk of actions; accept both shapes.
    if "action" in body:
        action = body["action"]
    elif "actions" in body and body["actions"]:
        action = body["actions"][0]
    else:
        raise AssertionError(f"VLA /predict response missing 'action(s)' key: {body}")
    assert isinstance(action, list), f"expected list, got {type(action)}"
    assert len(action) == 4, f"expected 4 floats, got {len(action)}"
    for i, val in enumerate(action):
        assert isinstance(val, (int, float)), f"action[{i}] is not numeric: {val!r}"
        assert math.isfinite(val), f"action[{i}]={val} is not finite"


def test_client_tool_single_step(live_url):
    class FakeCamera:
        def __init__(self, jpeg: bytes) -> None:
            self._jpeg = jpeg

        def capture_image(self) -> bytes:
            return self._jpeg

    jpeg = _synthetic_jpeg()
    env = FakeEnv(n_act=4)
    arm = ManiSkillArm(env)
    camera = FakeCamera(jpeg)

    cfg = VLAConfig(request_timeout=60.0)
    tool = create_vla_manipulation(
        tool_name="pick_cube_with_policy",
        tool_description=(
            "Execute a VLA-driven pick sequence by querying the VLA HTTP service."
        ),
        task_prompt="pick the cube",
        server_address=live_url,
        arm=arm,
        camera=camera,
        max_steps=1,
        cfg=cfg,
    )
    result = tool.invoke({})

    assert isinstance(result, str), f"expected str result, got {type(result)}"
    assert len(env.step_calls) >= 1, "arm received no step calls"
