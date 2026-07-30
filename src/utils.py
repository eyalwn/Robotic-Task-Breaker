"""Shared helpers used across robot modules and process entrypoints."""
from __future__ import annotations

import logging
import sys
from typing import TextIO

import config
from dotenv import find_dotenv, load_dotenv

_PROJECT_LOGGER_NAMES = (
    "config",
    "libero_env",
    "vla_client",
    "plan",
    "skills",
    "executor",
    "planner",
    "runner",
    "utils",
    "services",
    "__main__",
    "validation",
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
    logger trees are raised to ``level`` (arg or ``config.LOG_LEVEL``).
    Call from process entrypoints only — not on library import.
    """
    if level is None:
        level = config.LOG_LEVEL
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
