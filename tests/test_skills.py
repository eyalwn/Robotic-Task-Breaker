"""Skill catalog tests."""
from __future__ import annotations

from skills import arm_skill_catalog


def test_arm_skill_catalog_lists_core_arm_skills() -> None:
    catalog = arm_skill_catalog()
    for skill in (
        "arm.grab",
        "arm.place",
        "arm.transfer",
        "arm.open",
        "arm.close",
        "arm.push",
        "arm.pull",
    ):
        assert skill in catalog
