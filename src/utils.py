"""Logging setup for process entrypoints."""
from __future__ import annotations

import logging
import sys
import warnings
from typing import TextIO

import config

_PROJECT_LOGGER_NAMES = ("config", "envs", "vla_client", "plan", "skills",
                         "executor", "planner", "runner", "utils", "services",
                         "__main__")
_QUIET_LOGGER_NAMES = ("absl", "robosuite_logs")
_DEFAULT_FORMAT = "%(levelname)s [%(name)s] %(message)s"
_UPSTREAM_NOISE_SUBSTRINGS = (
    "No private macro file found", "It is recommended to use a private macro file",
    "To setup, run: python", "Could not import robosuite_models",
    "Could not load the mink-based whole-body IK",
    "mimicgen environments not imported since mimicgen is not installed")
_UPSTREAM_NOISE_LOGGER_PREFIXES = ("robosuite", "robocasa", "mimicgen")


def _upstream_noise_logger(name: str) -> bool:
    """True when ``name`` is a RoboCasa/RoboSuite upstream logger tree."""
    return (name == "robosuite_logs"
            or name.startswith(_UPSTREAM_NOISE_LOGGER_PREFIXES))


class _UpstreamNoiseFilter(logging.Filter):
    """Drop known harmless RoboCasa/robosuite optional-dep chatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not _upstream_noise_logger(record.name):
            return True
        message = record.getMessage()
        return not any(needle in message for needle in _UPSTREAM_NOISE_SUBSTRINGS)


def _pin_quiet_logger(name: str) -> None:
    """Hold this logger at ERROR.

    Robosuite sets ``robosuite_logs`` back to INFO on import.
    """
    logger = logging.getLogger(name)
    set_level = logging.Logger.setLevel.__get__(logger, logging.Logger)
    set_level(logging.ERROR)
    logger.setLevel = lambda level: set_level(logging.ERROR)


def _suppress_gymnasium_passive_obs_warnings() -> None:
    """GrootRoboCasaEnv dict obs fails PassiveEnvChecker; sim still runs."""
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=r"The obs returned by the .* method is not "
                            r"within the observation space\.",
                            module=r"gymnasium\.utils\.passive_env_checker")


def configure_logging(*, level: str = "", fmt: str = _DEFAULT_FORMAT,
                      stream: TextIO = sys.stdout) -> None:
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

    logging.basicConfig(level=logging.WARNING, format=fmt, stream=stream, force=True)
    noise_filter = _UpstreamNoiseFilter()
    for handler in logging.root.handlers:
        if (isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)):
            handler.setLevel(level)
            handler.addFilter(noise_filter)

    for name in _PROJECT_LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)
    for name in _QUIET_LOGGER_NAMES:
        _pin_quiet_logger(name)
    _suppress_gymnasium_passive_obs_warnings()
