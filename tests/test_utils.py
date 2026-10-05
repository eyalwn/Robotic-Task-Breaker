"""Logging setup keeps third-party sim noise off the task log."""
from __future__ import annotations

import logging
import sys

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
