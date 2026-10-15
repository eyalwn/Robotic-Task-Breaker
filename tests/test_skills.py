"""Skill catalog tests."""
from __future__ import annotations

from skills import skill_catalog


def test_skill_catalog_lists_runtime_skills() -> None:
    catalog = skill_catalog()
    for skill in ("open(", "close("):
        assert skill in catalog
    assert "object_name" in catalog
    for removed in ("place(", "put_in(", "turn_on(", "container_name"):
        assert removed not in catalog
    for legacy in ("transfer", "grab"):
        assert legacy not in catalog
