"""Scripted pick-cube policy tool (no ML, no server)."""
from __future__ import annotations

from langchain_core.tools import tool


def create_pick_cube_with_policy(arm):
    """Return a LangChain tool that runs a short scripted pick sequence."""

    @tool
    def pick_cube_with_policy() -> str:
        """Execute a scripted pick sequence: open gripper, lower, close, lift."""
        arm.set_gripper("open", steps=3)
        arm.move_ee(0.0, 0.0, -0.15, steps=5)
        arm.set_gripper("close", steps=5)
        arm.move_ee(0.0, 0.0, 0.2, steps=5)
        return "Scripted pick policy executed."

    return pick_cube_with_policy
