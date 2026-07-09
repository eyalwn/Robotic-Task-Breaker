"""RGB-based perception for LIBERO-10 task 0 (no sim ground truth)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.skills import DetectedObject, Pose, Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)

# LIBERO-10 task 0 manipulables (canonical MuJoCo body names + planner aliases).
TASK0_OBJECTS: dict[str, tuple[str, ...]] = {
    "alphabet_soup_1_main": (
        "alphabet soup can",
        "alphabet_soup",
        "alphabet_soup_1",
        "alphabet_soup_1_main",
        "soup can",
        "soup",
    ),
    "cream_cheese_1_main": (
        "cream cheese box",
        "cream_cheese",
        "cream_cheese_1",
        "cream_cheese_1_main",
        "cream cheese",
    ),
    "basket_1_main": (
        "basket",
        "basket_1",
        "basket_1_main",
    ),
}

# HSV ranges tuned for LIBERO agentview renders (OpenCV: H 0-179, S/V 0-255).
_COLOR_PROFILES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "alphabet_soup_1_main": ((0, 80, 80), (15, 255, 255)),
    "cream_cheese_1_main": ((90, 40, 80), (130, 255, 255)),
    "basket_1_main": ((10, 30, 40), (30, 180, 200)),
}


@dataclass(frozen=True)
class _Detection:
    body_name: str
    center_u: float
    center_v: float
    area: float
    confidence: float


def _normalize_object_name(name: str) -> str | None:
    raw = name.strip()
    if not raw:
        return None
    low = raw.lower()
    for body, aliases in TASK0_OBJECTS.items():
        if low == body.lower():
            return body
        if any(low == alias.lower() for alias in aliases):
            return body
    return None


def _jpeg_to_rgb(image_bytes: bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pixel_to_table_pose(u: float, v: float, width: int, height: int) -> Pose:
    """Map agentview pixel center to an approximate tabletop xyz."""
    nx = (u / max(width, 1) - 0.5) * 2.0
    ny = (v / max(height, 1) - 0.5) * 2.0
    x = 0.22 * nx
    y = -0.18 * ny
    z = 0.92
    return Pose(x=x, y=y, z=z)


def _detect_colored_blob(
    rgb,
    body_name: str,
    *,
    min_area: float = 120.0,
) -> _Detection | None:
    import cv2
    import numpy as np

    profile = _COLOR_PROFILES.get(body_name)
    if profile is None:
        return None
    lower, upper = profile
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area <= best_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        best_area = area
        best = (cx, cy)

    if best is None:
        return None
    cx, cy = best
    confidence = min(1.0, best_area / 2500.0)
    return _Detection(
        body_name=body_name,
        center_u=cx,
        center_v=cy,
        area=best_area,
        confidence=confidence,
    )


def detect_objects_in_frame(
    rgb,
    *,
    object_names: tuple[str, ...] | None = None,
) -> list[_Detection]:
    """Run task-0 CV heuristics on an RGB uint8 frame."""
    if rgb is None:
        return []
    targets = object_names or tuple(TASK0_OBJECTS)
    hits: list[_Detection] = []
    for body in targets:
        det = _detect_colored_blob(rgb, body)
        if det is not None:
            hits.append(det)
    hits.sort(key=lambda d: d.area, reverse=True)
    return hits


def _capture_rgb(camera) -> Any:
    capture = getattr(camera, "capture_image", None)
    if not callable(capture):
        logger.warning("real perception: camera has no capture_image()")
        return None
    try:
        jpeg = capture()
    except Exception as exc:
        logger.warning("real perception: capture_image failed: %s", exc)
        return None
    is_placeholder = getattr(camera, "is_placeholder_jpeg", None)
    if callable(is_placeholder) and is_placeholder(jpeg):
        logger.warning("real perception: placeholder frame — skipping detection")
        return None
    return _jpeg_to_rgb(jpeg)


class RealListObjectsSkill(Skill):
    name = "state.list_objects"
    description = (
        "Return manipulable object names detected in the current RGB view; "
        "empty when nothing is visible."
    )
    parameters: list[SkillParam] = []
    returns = "list[str]: object names"
    perception_source = "real"

    def __init__(
        self,
        camera,
        *,
        frame_fn: Callable[[], Any] | None = None,
    ) -> None:
        self._camera = camera
        self._frame_fn = frame_fn or (lambda: _capture_rgb(camera))

    def __call__(self, **kwargs: Any) -> list[str]:
        rgb = self._frame_fn()
        detections = detect_objects_in_frame(rgb)
        if detections:
            return [d.body_name for d in detections]
        if rgb is None:
            logger.warning(
                "state.list_objects: no camera frame — returning empty list"
            )
        else:
            logger.warning(
                "state.list_objects: no objects detected in frame — "
                "returning empty list"
            )
        return []


class RealDetectSkill(Skill):
    name = "perception.detect"
    description = "Detect a named object from the current RGB camera frame."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to detect"),
    ]
    returns = "DetectedObject | None"
    perception_source = "real"

    def __init__(
        self,
        camera,
        *,
        frame_fn: Callable[[], Any] | None = None,
    ) -> None:
        self._camera = camera
        self._frame_fn = frame_fn or (lambda: _capture_rgb(camera))

    def __call__(self, **kwargs: Any) -> DetectedObject | None:
        body = _normalize_object_name(str(kwargs.get("object_name", "")))
        if body is None:
            logger.debug("perception.detect: unknown object %r", kwargs.get("object_name"))
            return None
        rgb = self._frame_fn()
        if rgb is None:
            return None
        det = _detect_colored_blob(rgb, body)
        if det is None:
            return None
        h, w = rgb.shape[:2]
        pose = _pixel_to_table_pose(det.center_u, det.center_v, w, h)
        return DetectedObject(
            name=body,
            pose=pose,
            confidence=det.confidence,
        )


class RealLocateSkill(Skill):
    name = "perception.locate"
    description = "Return the 3-D position of a named object from RGB perception."
    parameters = [
        SkillParam("object_name", "str", "Name of the object to locate"),
    ]
    returns = "Pose | None"
    perception_source = "real"

    def __init__(
        self,
        camera,
        *,
        frame_fn: Callable[[], Any] | None = None,
    ) -> None:
        self._detect = RealDetectSkill(camera, frame_fn=frame_fn)

    def __call__(self, **kwargs: Any) -> Pose | None:
        result = self._detect(object_name=kwargs.get("object_name", ""))
        return result.pose if result is not None else None


def register_libero_real_perception(
    registry: SkillRegistry,
    camera,
    *,
    frame_fn: Callable[[], Any] | None = None,
) -> None:
    registry.register(RealListObjectsSkill(camera, frame_fn=frame_fn))
    registry.register_alias("list_objects", "state.list_objects")
    registry.register(RealDetectSkill(camera, frame_fn=frame_fn))
    registry.register(RealLocateSkill(camera, frame_fn=frame_fn))
