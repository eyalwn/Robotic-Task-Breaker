"""Logging setup keeps third-party sim noise off the task log."""
from __future__ import annotations

import logging
import sys
import warnings

from utils import configure_logging


def test_robosuite_warnings_stay_quiet_after_import_reset(capsys) -> None:
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    robosuite_logs = logging.getLogger("robosuite_logs")
    extra = logging.StreamHandler()
    try:
        configure_logging(level="DEBUG", stream=sys.stdout)
        robosuite_logs.addHandler(extra)
        robosuite_logs.setLevel(logging.INFO)
        robosuite_logs.warning("No private macro file found!")
        logging.getLogger("executor").info("executor-ok")
        captured = capsys.readouterr()
    finally:
        robosuite_logs.removeHandler(extra)
        root.handlers = previous_handlers
        root.setLevel(previous_level)

    assert "macro" not in captured.out
    assert "macro" not in captured.err
    assert "executor-ok" in captured.out


def test_upstream_optional_dep_noise_filtered(capsys) -> None:
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    robocasa = logging.getLogger("robocasa")
    mimicgen_msg = (
        "mimicgen environments not imported since mimicgen is not installed!"
    )
    try:
        configure_logging(level="DEBUG", stream=sys.stdout)
        robocasa.warning(mimicgen_msg)
        logging.getLogger("executor").warning(mimicgen_msg)
        logging.getLogger("runner").warning("No private macro file found!")
        captured = capsys.readouterr()
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)

    assert f"WARNING [executor] {mimicgen_msg}" in captured.out
    assert "WARNING [runner] No private macro file found!" in captured.out
    assert "WARNING [robocasa]" not in captured.out


def test_gymnasium_passive_obs_warning_filter_registered() -> None:
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    try:
        configure_logging(level="DEBUG", stream=sys.stdout)
        patterns = [
            item[1].pattern
            for item in warnings.filters
            if item[0] == "ignore" and item[1] is not None
        ]
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)

    assert any("observation space" in pattern for pattern in patterns)
