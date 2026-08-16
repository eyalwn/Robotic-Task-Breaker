"""HTTP client for the VLA policy service (``services/run_vla_server.py``).

Wire contract: ``POST /predict`` with multipart JPEG frames plus a JSON-encoded
state, answered with ``{"actions": [[...7 floats...], ...]}``. Backend-agnostic:
works with π0.5 (OpenPI) or SmolVLA depending on which service is running.
"""
from __future__ import annotations

import json
import logging
import math

import requests

import config

logger = logging.getLogger(__name__)

ACTION_DIM = 7


class VLAError(Exception):
    """The VLA service is unreachable or returned an unusable response."""


class VLAClient:
    """HTTP client for the VLA policy service."""

    def __init__(self, url: str = config.VLA_URL, timeout: float = config.VLA_TIMEOUT):
        self._url = url.rstrip("/")
        self._timeout = timeout

    def health(self) -> dict:
        """Query the service ``/health`` endpoint."""
        try:
            response = requests.get(f"{self._url}/health", timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise VLAError(f"VLA service GET /health failed: {exc}") from exc
        except ValueError as exc:
            raise VLAError(f"VLA service /health returned non-JSON: {exc}") from exc

    def predict(
        self, image: bytes, wrist: bytes, state: list[float], task: str
    ) -> list[list[float]]:
        """Return a chunk of 7-D actions for the current observation."""
        try:
            response = requests.post(
                f"{self._url}/predict",
                timeout=self._timeout,
                files={
                    "image": ("frame.jpg", image, "image/jpeg"),
                    "image_wrist": ("wrist.jpg", wrist, "image/jpeg"),
                },
                data={"task": task, "state": json.dumps(state)},
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise VLAError(f"VLA service POST /predict failed: {exc}") from exc
        except ValueError as exc:
            raise VLAError(f"VLA service /predict returned non-JSON: {exc}") from exc
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise VLAError(f"expected a non-empty 'actions' list, got {payload!r}")
        validated = [self._validate(action, idx) for idx, action in enumerate(actions)]
        return validated

    @staticmethod
    def _validate(action: object, idx: int) -> list[float]:
        if not isinstance(action, list) or len(action) != ACTION_DIM:
            raise VLAError(f"action {idx} must be {ACTION_DIM} floats, got {action!r}")
        try:
            values = [float(v) for v in action]
        except (TypeError, ValueError) as exc:
            raise VLAError(f"action {idx} is not numeric: {action!r}") from exc
        if not all(math.isfinite(v) for v in values):
            raise VLAError(f"action {idx} is not finite: {action!r}")
        clip = config.ACTION_CLIP
        clipped = [max(-clip, min(clip, v)) for v in values]
        if clipped != values:
            logger.warning("action %d exceeded +-%.1f: %s", idx, clip, values)
        return clipped
