from __future__ import annotations

import logging
from typing import Any

from .vision_health import VisionHealth

logger = logging.getLogger(__name__)

_PLACEHOLDER_CACHE: dict[str, bytes] = {}


def _frame_to_uint8(frame: Any):
    import numpy as np

    try:
        import torch

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
    except ImportError:
        pass

    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = (
            (arr * 255).clip(0, 255).astype(np.uint8)
            if arr.size and float(arr.max()) <= 1.0
            else arr.astype(np.uint8)
        )
    return arr


def _make_placeholder_jpeg(text: str) -> bytes:
    cached = _PLACEHOLDER_CACHE.get(text)
    if cached is not None:
        return cached

    import cv2
    import numpy as np

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img)
    data = buf.tobytes() if ok else b""
    _PLACEHOLDER_CACHE[text] = data
    return data


class SimCameraBase:
    _PLACEHOLDER_TEXT: str

    def __init__(self, env, arm=None) -> None:
        self._env = env
        self._arm = arm
        self._first_capture_path_logged = False
        self.vision_health = VisionHealth()
        self.vision_degraded = False
        self.session_unusable_for_vla = False
        self.last_capture_was_placeholder = False
        self._degraded_reason: str | None = None
        self._degraded_error_logged = False

    def set_arm(self, arm) -> None:
        self._arm = arm

    def _mark_vision_degraded(self, reason: str) -> None:
        self.vision_health.mark_unavailable(reason)
        self.vision_degraded = True
        self.session_unusable_for_vla = True
        if self._degraded_reason is None:
            self._degraded_reason = reason
        if self._degraded_error_logged:
            return
        self._degraded_error_logged = True
        self._log_vision_degraded(reason)

    def _log_vision_degraded(self, reason: str) -> None:
        logger.error("Vision degraded (%s).", reason)

    def is_placeholder_jpeg(self, image_bytes: bytes) -> bool:
        return image_bytes == _make_placeholder_jpeg(self._PLACEHOLDER_TEXT)

    def release(self) -> None:
        pass

    def reopen(self) -> None:
        pass

    def capture_image(self) -> bytes:
        raise NotImplementedError
