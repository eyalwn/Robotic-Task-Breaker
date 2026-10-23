"""Config constant tests."""
from __future__ import annotations

import config


def test_max_skill_steps_open_and_close() -> None:
    for skill in ("open", "close"):
        assert skill in config.MAX_SKILL_STEPS
        assert config.MAX_SKILL_STEPS[skill] > 0
    assert config.MAX_SKILL_STEPS["open"] == 720
    assert config.MAX_SKILL_STEPS["close"] == 400
    assert config.MAX_REPLANS == 2
