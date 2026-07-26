"""M7.4 direct VLA task probe (no planner/executor/LLM)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_path() -> None:
    """Allow `python validation/m7_direct_task_probe.py` to import `validation.*`."""
    if __package__ in (None, ""):
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)


_ensure_repo_root_on_path()

from validation.m7_wiring_probe import ensure_libero_config, sim_gt_fallback_guard

DEFAULT_SUITE = "libero_object"
DEFAULT_TASK_IDX = 0
DEFAULT_TASK_PROMPT = "pick up the object"
GATE = "m7.4"

logger = logging.getLogger(__name__)


def require_real_perception_backend() -> str:
    """Fail fast unless PERCEPTION_BACKEND resolves to real."""
    from sim.libero_perception import resolve_perception_backend

    mode = resolve_perception_backend()
    if mode != "real":
        raise ValueError(
            f"M7.4 requires PERCEPTION_BACKEND=real, got {mode!r}"
        )
    return mode


def evaluate_m7_4(
    *,
    payload: dict[str, Any] | None,
    completed_without_error: bool,
) -> tuple[str, str]:
    """Return (status, reason) for strict M7.4 env task-success gate."""
    if not completed_without_error:
        return "fail", "probe raised an exception"
    if payload is None:
        return "fail", "VLA tool returned non-JSON payload"

    reason = payload.get("reason")
    status = payload.get("status")
    if reason == "task_success" and status == "success":
        steps = payload.get("steps_applied")
        return "pass", f"env task success (steps_applied={steps})"
    if payload.get("task_success") is True and status == "success":
        steps = payload.get("steps_applied")
        return "pass", f"explicit task_success flag (steps_applied={steps})"

    if status == "success" and reason in {"grasp_lift", "proximity_grasp"}:
        return (
            "fail",
            f"geometry/grasp success reason={reason!r} rejected "
            "(M7.4 requires env task_success)",
        )
    return (
        "fail",
        f"expected reason=task_success status=success, "
        f"got reason={reason!r} status={status!r}",
    )


def _resolve_target_body(env: Any, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    from core.plan.presets import resolve_libero_grasp_object
    from sim.libero_skills import LiberoListObjectsSkill

    objects = LiberoListObjectsSkill(env)()
    return resolve_libero_grasp_object(objects)


def _close_env(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.warning("env.close() failed: %s", exc)


def run_direct_task_probe(
    *,
    task_suite: str,
    task_idx: int,
    task_prompt: str,
    target_body: str | None,
    server_url: str,
    seed: int | None,
) -> dict[str, Any]:
    """Run one direct VLA tool invocation and return metrics."""
    from sim.libero_backend import LiberoArm, LiberoCamera, create_env
    from vla.client import VLAClient, VLAServiceError
    from vla.config import VLAConfig
    from vla.tools import create_vla_manipulation

    os.environ.pop("VLA_AXIS_SIGNS", None)
    os.environ.pop("VLA_AXIS_REMAP", None)
    os.environ["VLA_USE_ENV_TASK_SUCCESS"] = "true"
    os.environ.setdefault("VLA_GRIPPER_MODE", "absolute_open_close")
    os.environ.setdefault("VLA_ACTION_SCALE", "1.0")
    os.environ.setdefault("LIBERO_EE_DELTA_SCALE", "1.0")
    os.environ.setdefault("LIBERO_VLA_IMAGE_FLIP", "1")
    os.environ.setdefault("LIBERO_OBSERVATION_HEIGHT", "256")
    os.environ.setdefault("LIBERO_OBSERVATION_WIDTH", "256")
    os.environ.setdefault("VLA_STEPS_PER_ACTION", "1")
    os.environ.setdefault("VLA_ADAPTIVE_SCALE", "false")
    os.environ["VLA_CHUNK_STEPS"] = "10"
    os.environ.setdefault("VLA_PROXIMITY_GRASP", "false")
    os.environ.setdefault("VLA_SIM_PICKCUBE_GRIPPER_PROFILE", "false")
    os.environ.setdefault("VLA_MAX_INNER_STEPS", "200")
    os.environ.setdefault("VLA_MAX_TOOL_SECONDS", "180")
    if seed is not None:
        os.environ["ROBOCREW_SIM_SEED"] = str(seed)
    ensure_libero_config()

    cfg = VLAConfig.from_env()
    client = VLAClient(server_url, timeout=cfg.request_timeout)
    client.health_check()

    payload: dict[str, Any] | None = None
    last_error: str | None = None
    resolved_target = target_body

    with sim_gt_fallback_guard():
        env = create_env(task_suite=task_suite, task_idx=task_idx)
        try:
            arm = LiberoArm(env)
            camera = LiberoCamera(env, arm=arm)
            resolved_target = _resolve_target_body(env, target_body)
            if not arm.set_target_object(resolved_target):
                raise RuntimeError(
                    f"could not set target object {resolved_target!r}"
                )

            vla_tool = create_vla_manipulation(
                tool_name="arm_vla_m7_direct_task",
                tool_description="M7.4 direct VLA task probe",
                task_prompt=task_prompt,
                server_address=server_url,
                arm=arm,
                camera=camera,
                max_steps=cfg.max_inner_steps,
                cfg=cfg,
            )
            raw = vla_tool.invoke({"task": task_prompt})
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.error("VLA tool returned non-JSON: %s", raw)
                payload = None
                last_error = "VLA tool returned non-JSON payload"
        except VLAServiceError as exc:
            last_error = str(exc)
            logger.error("VLA service error: %s", exc)
        finally:
            _close_env(env)

    return {
        "payload": payload,
        "task_prompt": task_prompt,
        "target_body": resolved_target,
        "seed": seed,
        "error": last_error,
        "completed_without_error": last_error is None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M7.4 direct VLA task probe (no planner decomposition)."
    )
    parser.add_argument("--task-suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-idx", type=int, default=DEFAULT_TASK_IDX)
    parser.add_argument(
        "--task-prompt",
        default=os.environ.get("M7_PROBE_TASK", DEFAULT_TASK_PROMPT),
    )
    parser.add_argument(
        "--target-body",
        default=os.environ.get("M7_DIRECT_TARGET", "").strip() or None,
        help="MuJoCo body to track (default: first graspable object in scene)",
    )
    parser.add_argument(
        "--vla-url",
        default=os.environ.get(
            "VLA_URL",
            f"http://localhost:{os.environ.get('VLA_PORT', '8765')}",
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="ROBOCREW_SIM_SEED override (default: env or 0)",
    )
    return parser.parse_args(argv)


def _print_final_result(
    *,
    status: str,
    reason: str,
    result: dict[str, Any] | None,
) -> None:
    label = "PASS" if status == "pass" else "FAIL"
    print(f"M7.4 probe {label}: {reason}")
    if result is None:
        return
    payload = result.get("payload") or {}
    print(
        f"reason={payload.get('reason')} status={payload.get('status')} "
        f"steps_applied={payload.get('steps_applied')} "
        f"target_body={result.get('target_body')} "
        f"task_prompt={result.get('task_prompt')!r}"
    )


def main(argv: list[str] | None = None) -> int:
    from utils import configure_logging, load_env

    load_env()
    args = parse_args(argv)

    configure_logging(stream=sys.stdout)

    seed = args.seed
    if seed is None:
        raw_seed = os.environ.get("ROBOCREW_SIM_SEED", "").strip()
        seed = int(raw_seed) if raw_seed else 0

    status = "fail"
    reason = "not started"
    result: dict[str, Any] | None = None

    try:
        require_real_perception_backend()
        logger.info(
            "M7.4 probe start: suite=%s task_idx=%s seed=%s prompt=%r",
            args.task_suite,
            args.task_idx,
            seed,
            args.task_prompt,
        )
        result = run_direct_task_probe(
            task_suite=args.task_suite,
            task_idx=args.task_idx,
            task_prompt=args.task_prompt,
            target_body=args.target_body,
            server_url=args.vla_url,
            seed=seed,
        )
        status, reason = evaluate_m7_4(
            payload=result.get("payload"),
            completed_without_error=result.get("completed_without_error", False),
        )
    except ValueError as exc:
        reason = str(exc)
        logger.error("M7.4 probe config error: %s", reason)
    except Exception:
        reason = traceback.format_exc(limit=12)
        logger.error("M7.4 probe failed:\n%s", reason)

    _print_final_result(status=status, reason=reason, result=result)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
