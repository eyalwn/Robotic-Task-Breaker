from __future__ import annotations

from typing import Protocol

from .schema import Plan


class Planner(Protocol):
    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        """Return a Plan for *goal* given the *catalog* of available skills.

        *context* carries replan context (e.g. feedback summary).
        """
        ...


class StaticPlanner:
    """Returns a preset plan (for handcrafted / no-LLM runs)."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        return self._plan
