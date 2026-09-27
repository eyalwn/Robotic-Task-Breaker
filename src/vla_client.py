"""HTTP client for the VLA policy service (``services/run_vla_server.py``).

Wire contract: ``POST /predict`` with multipart JPEG frames plus a JSON-encoded
state, answered with ``{"actions": [[...], ...]}``. Action vectors must be
finite; every vector in a chunk shares the same length (inferred from the first).
"""
from __future__ import annotations

import json
import logging
import math
import requests

import config

logger = logging.getLogger(__name__)


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
        self,
        images: tuple[bytes, ...],
        state: list[float],
        task: str,
    ) -> list[list[float]]:
        """Return a chunk of validated action vectors for the current observation."""
        if not images:
            raise VLAError("predict requires at least one image frame")
        try:
            response = requests.post(
                f"{self._url}/predict",
                timeout=self._timeout,
                files=[
                    ("images", (f"cam{i}.jpg", jpeg, "image/jpeg"))
                    for i, jpeg in enumerate(images)
                ],
                data={"task": task, "state": json.dumps(state)},
            )
            response.raise_for_status()
            payload: dict[str, list[list[float]]] = response.json()
            logger.debug("payload = %s", payload)
        except requests.RequestException as exc:
            raise VLAError(f"VLA service POST /predict failed: {exc}") from exc
        except ValueError as exc:
            raise VLAError(f"VLA service /predict returned non-JSON: {exc}") from exc
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise VLAError(f"expected a non-empty 'actions' list, got {payload!r}")
        return self._validate_chunk(actions)

    def _validate_chunk(self, actions: list[object]) -> list[list[float]]:
        out: list[list[float]] = []
        chunk_dim: int | None = None
        for idx, action in enumerate(actions):
            vec = self._parse_action(action, idx)
            if chunk_dim is None:
                chunk_dim = len(vec)
            elif len(vec) != chunk_dim:
                raise VLAError(
                    f"action {idx} length {len(vec)} != chunk length {chunk_dim}"
                )
            out.append(vec)
        return out

    @staticmethod
    def _parse_action(action: object, idx: int) -> list[float]:
        if not isinstance(action, list) or not action:
            raise VLAError(f"action {idx} must be a non-empty float list, got {action!r}")
        try:
            values = [float(v) for v in action]
        except (TypeError, ValueError) as exc:
            raise VLAError(f"action {idx} is not numeric: {action!r}") from exc
        if not all(math.isfinite(v) for v in values):
            raise VLAError(f"action {idx} is not finite: {action!r}")
        return values
