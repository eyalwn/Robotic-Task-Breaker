"""Unit tests for LIBERO VLA smoke helpers (no sim/VLA)."""
from __future__ import annotations

import sys
from pathlib import Path

_examples = Path(__file__).resolve().parents[1] / "examples"
if str(_examples) not in sys.path:
    sys.path.insert(0, str(_examples))

from libero_vla_smoke_common import grasp_passed  # noqa: E402


def test_grasp_passed_requires_confirmed_or_lift_reason() -> None:
    assert grasp_passed({"grasp_confirmed": True, "status": "stopped"})
    assert grasp_passed(
        {"reason": "grasp_lift", "status": "success", "grasp_confirmed": True}
    )
    assert not grasp_passed({"reason": "stagnation", "status": "stopped"})
    assert not grasp_passed({"reason": "grasp_lift", "status": "stopped"})


def test_sequential_smoke_prompts_match_libero_phrasing() -> None:
    from sim.libero_skills import libero_object_human_name

    soup = libero_object_human_name("alphabet_soup_1_main")
    cream = libero_object_human_name("cream_cheese_1_main")
    assert f"put the {soup} in the basket" == "put the alphabet soup can in the basket"
    assert f"pick up the {cream}" == "pick up the cream cheese box"
