"""Tests for services/run_vla_server.py policy dispatch."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import config
from run_vla_server import _SERVICE_FILES, main


def test_supported_policy_table_is_groot_only() -> None:
    assert _SERVICE_FILES == {"groot": "groot_service.py"}


def test_default_policy_is_groot_from_config() -> None:
    assert config.VLA_POLICY == "groot"
    with patch("run_vla_server.runpy.run_path") as mock_run:
        main()
    target = (
        Path(__file__).resolve().parent.parent / "services" / _SERVICE_FILES["groot"]
    )
    mock_run.assert_called_once_with(str(target), run_name="__main__")


def test_policy_groot_accepted() -> None:
    with patch.object(config, "VLA_POLICY", "groot"), patch(
        "run_vla_server.runpy.run_path"
    ) as mock_run:
        main()
    target = (
        Path(__file__).resolve().parent.parent / "services" / _SERVICE_FILES["groot"]
    )
    mock_run.assert_called_once_with(str(target), run_name="__main__")


def test_unknown_policy_exits_nonzero() -> None:
    with patch.object(config, "VLA_POLICY", "octo"), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
