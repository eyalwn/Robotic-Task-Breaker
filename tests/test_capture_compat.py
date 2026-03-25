"""Tests for capture_main_view (HW vs sim camera signatures)."""

from core.capture_compat import capture_main_view


def test_sim_style_camera_no_kwargs():
    class SimCam:
        def capture_image(self) -> bytes:
            return b"jpeg-sim"

    out = capture_main_view(
        SimCam(),
        camera_fov=90,
        navigation_mode="normal",
    )
    assert out == b"jpeg-sim"


def test_hw_style_camera_with_kwargs():
    class HWCam:
        def __init__(self) -> None:
            self.kwargs = None

        def capture_image(self, *, camera_fov=120, navigation_mode="normal") -> bytes:
            self.kwargs = (camera_fov, navigation_mode)
            return b"jpeg-hw"

    cam = HWCam()
    out = capture_main_view(cam, camera_fov=70, navigation_mode="precision")
    assert out == b"jpeg-hw"
    assert cam.kwargs == (70, "precision")
