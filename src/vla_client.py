"""HTTP client for the VLA policy service (``services/smolvla_service.py``).

Wire contract: ``POST /predict`` with multipart JPEG frames plus a JSON-encoded
state, answered with ``{"actions": [[...7 floats...], ...]}``.
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
    """HTTP client for the SmolVLA policy service."""

    def __init__(self, url: str = config.VLA_URL, timeout: float = config.VLA_TIMEOUT):
        self._url = url.rstrip("/")
        self._timeout = timeout

    def health(self) -> dict:
        """Query the service ``/health`` endpoint."""
        return self._request("GET", "/health")

    def predict(
        self, image: bytes, wrist: bytes, state: list[float], task: str
    ) -> list[list[float]]:
        """Return a chunk of 7-D actions for the current observation."""
        payload = self._request(
            "POST",
            "/predict",
            files={
                "image": ("frame.jpg", image, "image/jpeg"),
                "image_wrist": ("wrist.jpg", wrist, "image/jpeg"),
            },
            data={"task": task, "state": json.dumps(state)},
        )
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise VLAError(f"expected a non-empty 'actions' list, got {payload!r}")
        return [self._validate(action, idx) for idx, action in enumerate(actions)]

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

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = requests.request(
                method, f"{self._url}{path}", timeout=self._timeout, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise VLAError(f"VLA service {method} {path} failed: {exc}") from exc
        except ValueError as exc:
            raise VLAError(f"VLA service {path} returned non-JSON: {exc}") from exc
