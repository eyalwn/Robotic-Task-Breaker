"""Per-run execution trace: JSONL events written to a timestamped file."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _goal_slug(goal: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("_", goal.lower()).strip("_")
    return slug[:max_len]


class TraceWriter:
    """Appends JSONL trace events for one executor run.

    File name: ``{trace_dir}/{unix_timestamp_ms}_{goal_slug}.jsonl``
    """

    def __init__(self, trace_dir: str | Path, goal: str) -> None:
        path = Path(trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        fname = f"{ts}_{_goal_slug(goal)}.jsonl"
        self._path = path / fname
        self._file = self._path.open("a", encoding="utf-8")
        logger.debug("TraceWriter: %s", self._path)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, event: dict[str, Any]) -> None:
        event.setdefault("ts", time.time())
        try:
            self._file.write(json.dumps(event, default=str) + "\n")
            self._file.flush()
        except Exception as exc:
            logger.warning("TraceWriter.write failed: %s", exc)

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
