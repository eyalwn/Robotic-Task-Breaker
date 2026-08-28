"""Shared helpers used across robot modules and process entrypoints."""
from __future__ import annotations

import logging
import sys
from typing import TextIO

import config

_PROJECT_LOGGER_NAMES = (
    "config",
    "envs",
    "vla_client",
    "plan",
    "skills",
    "executor",
    "planner",
    "runner",
    "utils",
    "services",
    "__main__",
)
_QUIET_LOGGER_NAMES = (
    "absl",  # dm_control IK "Failed to converge" during VLABench stepping
)
_DEFAULT_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def configure_logging(
    *,
    level: str = "",
    extra_logger_names: tuple[str, ...] = (),
    fmt: str = _DEFAULT_FORMAT,
    stream: TextIO = sys.stdout,
    force: bool = True,
) -> None:
    """Configure root logging: project namespaces verbose, 3rd-party quiet.

    Root stays at WARNING so third-party INFO/DEBUG is suppressed. Longor
    logger trees are raised to ``level`` (arg or ``config.LOG_LEVEL``).
    Call from process entrypoints only — not on library import.
    """
    if not level:
        level = config.LOG_LEVEL
    resolved = getattr(logging, level.upper(), None)
    if not isinstance(resolved, int):
        raise ValueError(f"Invalid log level: {level!r}")
    level = resolved

    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        stream=stream,
        force=force,
    )
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)

    for name in _PROJECT_LOGGER_NAMES + extra_logger_names:
        logging.getLogger(name).setLevel(level)
    for name in _QUIET_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.ERROR)
