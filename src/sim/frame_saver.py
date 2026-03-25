"""Utility for saving observation frames to disk."""
from __future__ import annotations

import os
from pathlib import Path


class FrameSaver:
    """Writes JPEG frames to *output_dir* for debugging/proof."""

    def __init__(self, output_dir: str, save_every_n: int = 1, disabled: bool = False) -> None:
        self._disabled = disabled
        if not disabled:
            self._dir = Path(output_dir)
            self._dir.mkdir(parents=True, exist_ok=True)
        self._every = max(save_every_n, 1)

    def save(self, jpeg_bytes: bytes, loop_index: int) -> str | None:
        """Write *latest.jpg* every call and numbered frames on cadence.

        Returns the path of the numbered frame (*frame_XXXX.jpg*) when written,
        or ``None`` when only ``latest.jpg`` was updated.
        """
        if self._disabled:
            return None

        # Always update latest.jpg so callers can inspect the most recent frame.
        latest = self._dir / "latest.jpg"
        latest.write_bytes(jpeg_bytes)

        # On cadence, also write a numbered frame and return its path.
        if loop_index % self._every == 0:
            frame_path = self._dir / f"frame_{loop_index:04d}.jpg"
            frame_path.write_bytes(jpeg_bytes)
            return str(frame_path)

        return None

    def save_success(self, jpeg_bytes: bytes) -> str | None:
        """Write *pickcube_success.jpg* and return its path."""
        if self._disabled:
            return None
        path = self._dir / "pickcube_success.jpg"
        path.write_bytes(jpeg_bytes)
        return str(path)
