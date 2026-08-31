"""Skill catalog tests."""
from __future__ import annotations

from skills import skill_catalog


def test_skill_catalog_lists_core_skills() -> None:
    catalog = skill_catalog()
    for skill in ("grab", "place", "close", "turn_on"):
        assert skill in catalog
    assert "transfer" not in catalog
