from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class FeedbackDecision(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    REPLAN = "replan"


@dataclass(frozen=True)
class FeedbackOutcome:
    decision: FeedbackDecision
    reason: str = ""
    assessment: str | None = None


class FeedbackEvaluator(Protocol):
    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackOutcome:
        ...


class AlwaysContinueEvaluator:
    def evaluate(self, image_bytes: bytes, description: str, goal: str) -> FeedbackOutcome:
        return FeedbackOutcome(decision=FeedbackDecision.CONTINUE)
