"""Pure verdict logic for the RoboCasa M2 gate (hermetic; no sim or network)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

STAGE_A_EPISODES = 10
STAGE_A_PASS_MIN_SUCCESSES = 7
STAGE_B_TASK = "open_drawer"
STAGE_B_TASK_IDX = 0

StageStatus = Literal["PASS", "FAIL", "BLOCKED"]
OverallStatus = Literal["PASS", "FAIL", "BLOCKED"]

EPISODE_BOOL_LIST = re.compile(
    r"\[\s*(?:True|False)\s*(?:,\s*(?:True|False)\s*){9}\]"
)
SERVICE_RAW_SUCCESSES = re.compile(
    r"baseline successes=(\d+)/(\d+)"
)
LONGOR_RUN_LINE = re.compile(
    r"^(PASS|FAIL) task='open_drawer' task_idx=0 success=(True|False)"
)


@dataclass(frozen=True)
class StageVerdict:
    """Pass/fail/blocked outcome for one M2 gate stage."""

    name: str
    status: StageStatus
    detail: str
    successes: int | None = None
    episodes: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize this stage for ``verdict.json``."""
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }
        if self.successes is not None:
            payload["successes"] = self.successes
        if self.episodes is not None:
            payload["episodes"] = self.episodes
        return payload


@dataclass(frozen=True)
class M2GateVerdict:
    """Aggregate M2 gate outcome plus optional upstream reference metadata."""

    run_utc: str
    overall: OverallStatus
    detail: str
    stage_a: StageVerdict
    stage_b: StageVerdict
    upstream_primitive_reference: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the full gate document for ``verdict.json``."""
        payload: dict[str, object] = {
            "schema": "longor.robocasa_m2_gate.v1",
            "run_utc": self.run_utc,
            "overall": self.overall,
            "detail": self.detail,
            "stage_a_adapter_parity": self.stage_a.to_dict(),
            "stage_b_longor_slice": self.stage_b.to_dict(),
        }
        if self.upstream_primitive_reference is not None:
            payload["upstream_primitive_reference"] = self.upstream_primitive_reference
        return payload


def utc_run_id(when: datetime | None = None) -> str:
    """Return a timestamp string for artifact directory names."""
    moment = when or datetime.now(timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def parse_upstream_rollout_log(log_text: str) -> tuple[int | None, str, list[bool] | None]:
    """Parse M1-style ``results:`` line with a bounded 10-bool episode list."""
    for line in reversed(log_text.splitlines()):
        if not line.startswith("results:"):
            continue
        match = EPISODE_BOOL_LIST.search(line)
        if not match:
            return None, "results line lacks bounded 10-bool episode list", None
        tokens = re.findall(r"\b(True|False)\b", match.group(0))
        if len(tokens) != STAGE_A_EPISODES:
            return (
                None,
                f"episode list token count {len(tokens)} != {STAGE_A_EPISODES}",
                None,
            )
        bools = [token == "True" for token in tokens]
        return sum(bools), "parsed bounded 10-bool list on results line", bools
    return None, "no results: line in rollout.log", None


def upstream_gate_verdict_line(reference: dict[str, object]) -> str:
    """Format an M1-style ``verdict.txt`` line from upstream reference metadata."""
    status = reference.get("status")
    detail = str(reference.get("detail", ""))
    if status == "PASS":
        prefix = "GATE_PASS"
    elif status == "FAIL":
        prefix = "GATE_FAIL"
    else:
        prefix = "GATE_BLOCKED"
    return f"{prefix} {detail}".strip()


def upstream_reference_from_rollout_log(log_path: Path) -> dict[str, object]:
    """Build metadata from an upstream primitive gate rollout log."""
    if not log_path.is_file():
        return {
            "status": "BLOCKED",
            "detail": f"missing rollout log: {log_path}",
            "path": str(log_path),
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    successes, note, episode_successes = parse_upstream_rollout_log(text)
    if successes is None:
        return {
            "status": "BLOCKED",
            "detail": note,
            "path": str(log_path),
        }
    status: StageStatus = (
        "PASS" if successes >= STAGE_A_PASS_MIN_SUCCESSES else "FAIL"
    )
    return {
        "status": status,
        "detail": f"successes={successes}/{STAGE_A_EPISODES} ({note})",
        "successes": successes,
        "episodes": STAGE_A_EPISODES,
        "episode_successes": episode_successes,
        "path": str(log_path),
    }


def stage_a_from_service_successes(successes: int, episodes: int) -> StageVerdict:
    """Map service-backed raw OpenDrawer successes to stage A verdict."""
    if episodes != STAGE_A_EPISODES:
        return StageVerdict(
            name="stage_a_adapter_parity",
            status="BLOCKED",
            detail=f"expected {STAGE_A_EPISODES} episodes, got {episodes}",
            successes=successes,
            episodes=episodes,
        )
    if successes >= STAGE_A_PASS_MIN_SUCCESSES:
        status: StageStatus = "PASS"
    else:
        status = "FAIL"
    return StageVerdict(
        name="stage_a_adapter_parity",
        status=status,
        detail=f"service-backed raw OpenDrawer successes={successes}/{episodes}",
        successes=successes,
        episodes=episodes,
    )


def stage_a_blocked(reason: str) -> StageVerdict:
    """Return a blocked stage A verdict."""
    return StageVerdict(
        name="stage_a_adapter_parity",
        status="BLOCKED",
        detail=reason,
    )


def stage_b_from_runner_success(success: bool) -> StageVerdict:
    """Map one frozen OpenDrawer Runner episode to stage B verdict."""
    status: StageStatus = "PASS" if success else "FAIL"
    return StageVerdict(
        name="stage_b_longor_slice",
        status=status,
        detail=(
            f"Runner task={STAGE_B_TASK!r} task_idx={STAGE_B_TASK_IDX} "
            f"success={success}"
        ),
    )


def stage_b_blocked(reason: str) -> StageVerdict:
    """Return a blocked stage B verdict."""
    return StageVerdict(
        name="stage_b_longor_slice",
        status="BLOCKED",
        detail=reason,
    )


def parse_service_raw_stdout(text: str) -> tuple[int | None, int | None, str]:
    """Parse ``run_robocasa_raw_policy.py`` aggregate stdout."""
    for line in reversed(text.splitlines()):
        match = SERVICE_RAW_SUCCESSES.search(line)
        if not match:
            continue
        successes = int(match.group(1))
        episodes = int(match.group(2))
        return successes, episodes, "parsed baseline successes line"
    return None, None, "no baseline successes line in service raw stdout"


def parse_longor_runner_stdout(text: str) -> tuple[bool | None, str]:
    """Parse ``run_longor_task.py`` stdout for the frozen OpenDrawer episode."""
    for line in reversed(text.splitlines()):
        match = LONGOR_RUN_LINE.match(line.strip())
        if match:
            success = match.group(2) == "True"
            return success, f"parsed {match.group(1)} runner line"
        if line.startswith("FAIL task='open_drawer'"):
            return False, f"runner failed: {line.strip()}"
    return None, "no OpenDrawer runner verdict line in stdout"


def combine_overall(stage_a: StageVerdict, stage_b: StageVerdict) -> tuple[OverallStatus, str]:
    """Compute overall M2 gate status from both stages."""
    if stage_a.status == "BLOCKED" or stage_b.status == "BLOCKED":
        reasons = []
        if stage_a.status == "BLOCKED":
            reasons.append(f"stage_a: {stage_a.detail}")
        if stage_b.status == "BLOCKED":
            reasons.append(f"stage_b: {stage_b.detail}")
        return "BLOCKED", "; ".join(reasons)
    if stage_a.status == "PASS" and stage_b.status == "PASS":
        return "PASS", "stage_a and stage_b passed"
    parts = []
    if stage_a.status == "FAIL":
        parts.append(f"stage_a: {stage_a.detail}")
    if stage_b.status == "FAIL":
        parts.append(f"stage_b: {stage_b.detail}")
    return "FAIL", "; ".join(parts) if parts else "stage failure"


def build_verdict(
    run_utc: str,
    stage_a: StageVerdict,
    stage_b: StageVerdict,
    upstream_primitive_reference: dict[str, object] | None = None,
) -> M2GateVerdict:
    """Assemble the aggregate gate document."""
    overall, detail = combine_overall(stage_a, stage_b)
    return M2GateVerdict(
        run_utc=run_utc,
        overall=overall,
        detail=detail,
        stage_a=stage_a,
        stage_b=stage_b,
        upstream_primitive_reference=upstream_primitive_reference,
    )


def write_verdict_artifacts(artifact_dir: Path, verdict: M2GateVerdict) -> None:
    """Write ``verdict.json`` and ``verdict.txt`` under ``artifact_dir``."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = verdict.to_dict()
    json_path = artifact_dir / "verdict.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    txt_line = (
        f"M2_GATE_{verdict.overall} {verdict.detail} "
        f"(stage_a={verdict.stage_a.status}, stage_b={verdict.stage_b.status})"
    )
    (artifact_dir / "verdict.txt").write_text(txt_line + "\n", encoding="utf-8")
