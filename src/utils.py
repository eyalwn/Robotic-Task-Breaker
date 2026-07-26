"""Shared helpers used across robot modules and process entrypoints."""
from __future__ import annotations

import functools
import logging
import math
import os
import sys
from typing import TextIO

from dotenv import find_dotenv, load_dotenv

_PROJECT_LOGGER_NAMES = (
    "core",
    "sim",
    "vla",
    "validation",
    "services",
    "robots",
    "utils",
    "__main__",
)
_DEFAULT_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def load_env(*, usecwd: bool = False) -> None:
    """Load repo ``.env`` into ``os.environ`` (does not override existing keys).

    Call from process entrypoints only — not on library import. Prefer calling
    before ``configure_logging`` so ``LOG_LEVEL`` from ``.env`` is visible.
    """
    load_dotenv(find_dotenv(usecwd=usecwd))


def configure_logging(
    *,
    level: str | int | None = None,
    extra_logger_names: tuple[str, ...] = (),
    fmt: str = _DEFAULT_FORMAT,
    stream: TextIO | None = None,
    force: bool = True,
) -> None:
    """Configure root logging: project namespaces verbose, 3rd-party quiet.

    Root stays at WARNING so third-party INFO/DEBUG is suppressed. Longor
    logger trees are raised to ``level`` (arg or ``LOG_LEVEL``, default INFO).
    Call from process entrypoints only — not on library import.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        resolved = getattr(logging, level.upper(), None)
        if not isinstance(resolved, int):
            raise ValueError(f"Invalid log level: {level!r}")
        level = resolved

    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        stream=sys.stdout if stream is None else stream,
        force=force,
    )
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)

    for name in _PROJECT_LOGGER_NAMES + extra_logger_names:
        logging.getLogger(name).setLevel(level)


def basic_augmentation(image, h_fov=120, center_angle=0, navigation_mode="normal"):
    """Draw horizontal angle markers on the bottom of the image."""
    import cv2
    height, width = image.shape[:2]
    yellow = (0, 255, 255)
    orange = (0, 100, 255)
    y_pos = 25
    mark_len_angle = 10

    # Draw baseline
    cv2.line(image, (0, y_pos), (width, y_pos), yellow, 2)

    # Generate markers every 5° within visible range
    nr_of_marks = int((h_fov / 2) // mark_len_angle * 2 + 1)
    pixels_per_mark = width / h_fov * mark_len_angle
    start_pixel = (width - (nr_of_marks - 1) * pixels_per_mark) / 2
    start_angle = (-h_fov / 2 + center_angle)
    start_angle = mark_len_angle * math.trunc(start_angle / mark_len_angle)

    for mark_number in range(nr_of_marks):
        x = int(start_pixel + mark_number * pixels_per_mark)
        angle = start_angle + mark_number * mark_len_angle
        cv2.line(image, (x, y_pos - 10), (x, y_pos + 10), orange, 2)
        cv2.putText(image, f"{angle}", (x - 15, y_pos + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, orange, 2)

    # put right/left text
    cv2.putText(image, "<=LEFT", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, orange, 2)
    cv2.putText(image, "RIGHT=>", (width - 145, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)

    # range of arms horizontalline
    if navigation_mode == "precision":
        image = draw_precision_mode_aug(image, width, height)
    return image


def draw_precision_mode_aug(image, width, height):
    import cv2
    # draw arms range lines
    cv2.line(image, (int(width*0.15), int(0.40*height)), (int(width*0.30), int(0.28*height)), (0, 255, 0), 4)
    cv2.line(image, (int(width*0.30), int(0.28*height)), (int(width*0.70), int(0.28*height)), (0, 255, 0), 4)
    cv2.line(image, (int(width*0.70), int(0.28*height)), (int(width*0.85), int(0.40*height)), (0, 255, 0), 4)
    cv2.putText(image, "arm range", (int(width*0.65), int(0.28*height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # body continuation lines
    cv2.line(image, (int(width*0.22), int(height*0.75)),
                    (int(width*0.27), int(height*0.42)), (0, 165, 255), 2)
    cv2.line(image, (int(width*0.78), int(height*0.75)),
                    (int(width*0.73), int(height*0.42)), (0, 165, 255), 2)
    return image


augment_image = basic_augmentation


def stop_listening_during_tool_execution(sound_receiver):
    """
    Decorator to stop listening before function execution and resume after.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if sound_receiver is not None:
                sound_receiver.stop_listening()
            result = func(*args, **kwargs)
            if sound_receiver is not None:
                sound_receiver.start_listening()
            return result
        return wrapper
    return decorator
