from __future__ import annotations

from enum import Enum
from typing import Protocol


class FeedbackDecision(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    REPLAN = "replan"


class FeedbackEvaluator(Protocol):
    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackDecision:
        ...
