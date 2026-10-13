"""Skill catalog tests."""
from __future__ import annotations

from skills import skill_catalog


def test_skill_catalog_lists_runtime_skills() -> None:
    catalog = skill_catalog()
    for skill in ("open(", "place(", "put_in(", "close(", "turn_on("):
        assert skill in catalog
    assert "object_name" in catalog
    assert "container_name" in catalog
    assert "turn_on(target_name:" in catalog
    for legacy in ("transfer", "grab"):
        assert legacy not in catalog
