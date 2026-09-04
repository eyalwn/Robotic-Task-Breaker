"""Tests for services/run_vla_server.py policy dispatch."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import config
from run_vla_server import _SERVICE_FILES, main


def test_supported_policy_table_is_pi0fast_and_smolvla() -> None:
    assert set(_SERVICE_FILES) == {"pi0fast", "smolvla"}


def test_default_policy_is_pi0fast_from_config() -> None:
    assert config.VLA_POLICY == "pi0fast"
    with patch("run_vla_server.runpy.run_path") as mock_run:
        main()
    target = (
        Path(__file__).resolve().parent.parent / "services" / _SERVICE_FILES["pi0fast"]
    )
    mock_run.assert_called_once_with(str(target), run_name="__main__")


def test_policy_smolvla_accepted() -> None:
    with patch.object(config, "VLA_POLICY", "smolvla"), patch(
        "run_vla_server.runpy.run_path"
    ) as mock_run:
        main()
    target = (
        Path(__file__).resolve().parent.parent / "services" / _SERVICE_FILES["smolvla"]
    )
    mock_run.assert_called_once_with(str(target), run_name="__main__")


def test_unknown_policy_exits_nonzero() -> None:
    with patch.object(config, "VLA_POLICY", "octo"), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
