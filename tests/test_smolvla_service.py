"""Hermetic tests for SmolVLA service wire-format helpers."""
from __future__ import annotations

import numpy as np
import pytest

import config
from smolvla_service import actions_to_wire_format, truncate_action_chunk


def test_truncates_to_max_actions() -> None:
    chunk = np.arange(30, dtype=np.float32).reshape(10, 3)
    truncated = truncate_action_chunk(chunk, 3)
    assert truncated.shape == (3, 3)
    assert truncated[0, 0] == 0.0
    assert truncated[2, 0] == 6.0


def test_truncates_when_chunk_equals_max() -> None:
    chunk = np.ones((5, 7), dtype=np.float32)
    truncated = truncate_action_chunk(chunk, 5)
    assert truncated.shape == (5, 7)


def test_truncates_when_chunk_smaller_than_max() -> None:
    chunk = np.ones((2, 7), dtype=np.float32)
    truncated = truncate_action_chunk(chunk, 5)
    assert truncated.shape == (2, 7)


def test_truncates_empty_chunk() -> None:
    chunk = np.zeros((0, 7), dtype=np.float32)
    truncated = truncate_action_chunk(chunk, 5)
    assert truncated.shape == (0, 7)


def test_truncates_rejects_non_positive_max() -> None:
    chunk = np.ones((2, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="max_actions must be positive"):
        truncate_action_chunk(chunk, 0)


def test_truncates_rejects_non_2d_chunk() -> None:
    chunk = np.ones((2, 3, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 2D actions"):
        truncate_action_chunk(chunk, 5)


def test_wire_format_truncated_to_config_limit() -> None:
    chunk = np.full((10, 7), 3.0, dtype=np.float32)
    truncated = truncate_action_chunk(chunk, config.VLA_ACTIONS_PER_CHUNK)
    wire = actions_to_wire_format(truncated)
    assert len(wire) == config.VLA_ACTIONS_PER_CHUNK
    assert all(len(action) == 7 for action in wire)
    assert wire[0][0] == config.ACTION_CLIP
