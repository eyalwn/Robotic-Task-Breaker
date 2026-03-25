from __future__ import annotations

import logging
import math
from typing import List

import requests


logger = logging.getLogger(__name__)
DEFAULT_ACTION_CLIP = 2.0


class VLAServiceError(Exception):
    """Raised when the VLA HTTP service is unreachable or returns a bad response."""


class VLAClient:
    """HTTP client for a VLA inference service."""

    def __init__(self, service_url: str, timeout: float = 60.0):
        self._service_url = service_url.rstrip("/")
        self._timeout = timeout

    @staticmethod
    def _normalise_action(action: list[float], idx: int) -> list[float]:
        if len(action) != 4:
            raise VLAServiceError(
                f"VLA action at index {idx} must have 4 elements, got {len(action)}"
            )
        try:
            values = [float(v) for v in action]
        except (TypeError, ValueError) as exc:
            raise VLAServiceError(
                f"VLA action at index {idx} contains non-numeric values: {action!r}"
            ) from exc
        if not all(math.isfinite(v) for v in values):
            raise VLAServiceError(
                f"VLA action at index {idx} contains non-finite values: {action!r}"
            )
        clipped = [max(-DEFAULT_ACTION_CLIP, min(DEFAULT_ACTION_CLIP, v)) for v in values]
        if clipped != values:
            logger.warning(
                "VLA action at index %d exceeded range [-%.1f, %.1f]; clipped %s -> %s",
                idx,
                DEFAULT_ACTION_CLIP,
                DEFAULT_ACTION_CLIP,
                values,
                clipped,
            )
        return clipped

    def predict(self, image_bytes: bytes, task: str) -> List[List[float]]:
        """POST image to /predict, return action chunk (list of [dx,dy,dz,gripper])."""
        try:
            response = requests.post(
                f"{self._service_url}/predict",
                files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
                data={"task": task},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise VLAServiceError(f"VLA service connection error: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise VLAServiceError(f"VLA service error: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise VLAServiceError(f"VLA service returned non-JSON response: {exc}") from exc

        if "actions" in payload:
            chunk = payload["actions"]
        elif "action" in payload:
            chunk = [payload["action"]]
        else:
            logger.error("VLA service returned unexpected payload: %s", payload)
            raise VLAServiceError(f"Unexpected VLA response payload: {payload!r}")

        if not isinstance(chunk, list):
            raise VLAServiceError(f"VLA actions payload must be a list, got {type(chunk)}")

        normalised: List[List[float]] = []
        for idx, action in enumerate(chunk):
            if not isinstance(action, list):
                raise VLAServiceError(
                    f"VLA action at index {idx} must be a list, got {type(action)}"
                )
            normalised.append(self._normalise_action(action, idx))

        return normalised

    def health_check(self) -> dict:
        """GET /health, return parsed JSON. Raise on failure."""
        try:
            response = requests.get(
                f"{self._service_url}/health",
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise VLAServiceError(f"VLA health check failed: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise VLAServiceError(f"VLA health check returned non-JSON response: {exc}") from exc

