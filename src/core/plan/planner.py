from __future__ import annotations

from typing import Protocol

from .schema import Plan


class Planner(Protocol):
    def plan(self, goal: str, catalog: str, context: str = "") -> Plan:
        """Return a Plan for *goal* given the *catalog* of available skills.

        *context* carries replan context (e.g. feedback summary). Real LLM
        call deferred to M6.
        """
        ...
