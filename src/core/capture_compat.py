"""Helpers for cameras with different `capture_image` signatures (HW vs sim)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CameraJPEG(Protocol):
    """Minimal contract: return one JPEG frame as bytes."""

    def capture_image(self, *args: Any, **kwargs: Any) -> bytes: ...


def capture_main_view(
    camera: CameraJPEG,
    *,
    camera_fov: float | int,
    navigation_mode: str,
) -> bytes:
    """Call `capture_image` with HW kwargs when supported; sim cameras use no extra args."""
    try:
        return camera.capture_image(
            camera_fov=camera_fov,
            navigation_mode=navigation_mode,
        )
    except TypeError:
        return camera.capture_image()
