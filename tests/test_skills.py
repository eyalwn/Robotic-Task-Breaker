"""Skill catalog tests."""
from __future__ import annotations

from skills import skill_catalog


def test_skill_catalog_lists_open_only() -> None:
    catalog = skill_catalog()
    assert "open(" in catalog
    assert "transfer" not in catalog
    for skill in ("grab", "place", "close", "turn_on"):
        assert skill not in catalog
