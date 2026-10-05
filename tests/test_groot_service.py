"""Hermetic tests for GR00T service wire-format helpers."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from envs.robocasa_env import (
    ACTION_DIMS,
    ACTION_KEYS,
    STATE_KEYS,
    expected_action_dim,
    expected_state_dim,
    unflatten_state,
)
from groot_service import (
    GROOT_LANGUAGE_KEY,
    actions_to_wire_format,
    build_groot_observation,
    decode_image,
    groot_action_dict_to_chunk,
    parse_state_field,
    truncate_action_chunk,
)


def _state_json() -> str:
    return json.dumps([float(i) for i in range(expected_state_dim())])


def _jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_image_accepts_png_bytes() -> None:
    arr = decode_image(_png_bytes())
    assert arr.shape == (256, 256, 3)
    assert arr.dtype == np.uint8


def test_parse_state_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="state must have"):
        parse_state_field(json.dumps([0.0, 1.0]))


def test_parse_state_rejects_non_finite() -> None:
    bad = [0.0] * expected_state_dim()
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        parse_state_field(json.dumps(bad))


def test_unflatten_state_roundtrip_via_build_observation() -> None:
    flat = [float(i) for i in range(expected_state_dim())]
    side = np.zeros((256, 256, 3), dtype=np.uint8)
    obs = build_groot_observation(
        side, side, side, "Open the drawer", np.asarray(flat))
    assert GROOT_LANGUAGE_KEY in obs
    assert obs[GROOT_LANGUAGE_KEY] == ("Open the drawer",)
    for key in STATE_KEYS:
        assert f"state.{key}" in obs
    groups = unflatten_state(flat)
    assert groups["gripper_qpos"] == [0.0, 1.0]


def test_groot_action_dict_to_chunk_and_wire() -> None:
    horizon = 2
    action_dict = {}
    for key, width in zip(ACTION_KEYS, ACTION_DIMS):
        data = np.arange(
            horizon * width, dtype=np.float32).reshape(1, horizon, width)
        action_dict[f"action.{key}"] = data
    chunk = groot_action_dict_to_chunk(action_dict)
    assert chunk.shape == (horizon, expected_action_dim())
    wire = actions_to_wire_format(truncate_action_chunk(chunk, 1))
    assert len(wire) == 1
    assert len(wire[0]) == expected_action_dim()


def test_actions_to_wire_rejects_bad_dim() -> None:
    chunk = np.zeros((1, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="expected action dim"):
        actions_to_wire_format(chunk)
