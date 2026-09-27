"""Skill catalog tests."""
from __future__ import annotations

from skills import skill_catalog


def test_skill_catalog_lists_m2_and_transfer() -> None:
    catalog = skill_catalog()
    assert "open_drawer" in catalog
    assert "transfer" in catalog
    for skill in ("grab", "place", "close", "turn_on"):
        assert skill not in catalog
