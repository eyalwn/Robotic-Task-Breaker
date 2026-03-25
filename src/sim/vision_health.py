from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisionHealth:
    """Single source of truth for sim vision reliability."""

    last_frame_real: bool = False
    last_capture_path: str | None = None
    degraded_reason: str | None = None
    session_unusable_for_vla: bool = False

    def mark_real(self, path: str) -> None:
        self.last_frame_real = True
        self.last_capture_path = path

    def mark_unavailable(self, reason: str) -> None:
        self.last_frame_real = False
        self.degraded_reason = self.degraded_reason or reason
        self.session_unusable_for_vla = True

    @property
    def vision_unavailable(self) -> bool:
        return self.session_unusable_for_vla or (not self.last_frame_real)
