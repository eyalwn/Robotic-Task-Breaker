"""Writes LIBERO arm-step frames to an mp4 file.

Activated by setting LONGOR_VIDEO_OUT=<path>. No-ops when env var is unset.
"""
from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(self, path: str, fps: int = 10) -> None:
        self._path = Path(path)
        self._fps = fps
        self._writer = None
        self._closed = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("LIBERO video recording enabled: %s", self._path)
        atexit.register(self.close)

    def add_frame(self, img: "np.ndarray") -> None:
        import numpy as np

        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
            return

        import cv2

        if self._writer is None:
            h, w = arr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._path), fourcc, self._fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter failed to open {self._path}")

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("video saved: %s", self._path)
