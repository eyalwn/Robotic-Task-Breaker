"""Config constant tests."""
from __future__ import annotations

import config


def test_max_skill_steps_includes_put_in_and_close() -> None:
    for skill in ("open", "place", "put_in", "close", "turn_on"):
        assert skill in config.MAX_SKILL_STEPS
        assert config.MAX_SKILL_STEPS[skill] > 0
    assert config.MAX_SKILL_STEPS["put_in"] == 720
    assert config.MAX_SKILL_STEPS["close"] == 400
