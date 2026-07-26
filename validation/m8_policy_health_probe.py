"""M8 isolated SmolVLA policy-health probe (same isolation as M7.2)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_path() -> None:
    """Allow `python validation/m8_policy_health_probe.py` imports."""
    if __package__ in (None, ""):
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)


_ensure_repo_root_on_path()

from validation.m7_wiring_probe import ensure_libero_config

DEFAULT_SUITE = "libero_object"
DEFAULT_TASK_IDX = 0
DEFAULT_TASK_PROMPT = "pick up the alphabet soup can"
DEFAULT_MAX_STEPS = 120
DEFAULT_APPROACH_DIST = 0.12
DEFAULT_MIN_DIST_DROP = 0.06
DEFAULT_MIN_IMPROVING_STEPS = 8
DEFAULT_MIN_LIFT_DELTA = 0.02
DEFAULT_GRIPPER_CLOSE_THRESHOLD = 0.5

_DIST_EPS = 1e-6

logger = logging.getLogger(__name__)


def _dist(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) < 3 or len(b) < 3:
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])))


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return bool(value.flat[0])
    except Exception:
        pass
    return bool(value)


def _env_task_success(arm: Any) -> bool:
    try:
        env = getattr(arm, "_env", None)
        raw = getattr(env, "unwrapped", env)
        if raw is not None:
            info = raw.evaluate()
            if isinstance(info, dict) and _as_bool(info.get("success")):
                return True
    except Exception:
        pass

    info = getattr(arm, "last_info", {}) or {}
    for key in ("success", "is_success", "task_success"):
        if _as_bool(info.get(key)):
            return True
    return False


def _close_env(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.warning("env.close() failed: %s", exc)


def _resolve_target_body(env: Any, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    from core.plan.presets import resolve_libero_grasp_object
    from sim.libero_skills import LiberoListObjectsSkill

    objects = LiberoListObjectsSkill(env)()
    return resolve_libero_grasp_object(objects)


def _apply_m8_env_defaults(seed: int | None) -> None:
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
    # M8 follows M7.2 isolation doctrine: closed-loop n_action_steps=1.
    os.environ["VLA_CHUNK_STEPS"] = "1"
    os.environ.setdefault("VLA_PROXIMITY_GRASP", "false")
    os.environ.setdefault("VLA_SIM_PICKCUBE_GRIPPER_PROFILE", "false")
    if seed is not None:
        os.environ["ROBOCREW_SIM_SEED"] = str(seed)


def _apply_action(arm: Any, cfg: Any, raw_action: list[float]) -> float:
    from vla.action import process_raw_action

    clipped = [max(-cfg.action_clip, min(cfg.action_clip, float(v))) for v in raw_action]
    if len(clipped) == 7:
        arm.apply_vla_action(
            clipped[0],
            clipped[1],
            clipped[2],
            clipped[3],
            clipped[4],
            clipped[5],
            clipped[6],
            steps=cfg.steps_per_action,
        )
        return clipped[6]
    if len(clipped) == 4:
        dx, dy, dz = process_raw_action(clipped[:3], cfg)
        arm.move_ee(dx, dy, dz, steps=cfg.steps_per_action)
        return clipped[3]
    raise ValueError(f"VLA action must have 4 or 7 elements, got {len(clipped)}")


def run_m8_policy_health_probe(
    *,
    task_suite: str,
    task_idx: int,
    task_prompt: str,
    target_body: str | None,
    server_url: str,
    seed: int | None,
    max_steps: int,
    min_lift_delta: float,
    gripper_close_threshold: float,
) -> dict[str, Any]:
    """Run one isolated policy-health episode and return detailed metrics."""
    from sim.libero_backend import LiberoArm, LiberoCamera, create_env
    from vla.client import VLAClient, VLAServiceError
    from vla.config import VLAConfig

    _apply_m8_env_defaults(seed)
    ensure_libero_config()

    cfg = VLAConfig.from_env()
    client = VLAClient(server_url, timeout=cfg.request_timeout)
    client.health_check()

    env = None
    error: str | None = None
    resolved_target = target_body
    task_success = False
    lift_step: int | None = None
    success_step: int | None = None
    steps_applied = 0

    initial_dist: float | None = None
    best_dist: float | None = None
    improving_steps = 0
    close_steps = 0
    max_lift = 0.0
    trace: list[dict[str, Any]] = []

    try:
        env = create_env(task_suite=task_suite, task_idx=task_idx)
        arm = LiberoArm(env)
        camera = LiberoCamera(env, arm=arm)

        resolved_target = _resolve_target_body(env, target_body)
        if not arm.set_target_object(resolved_target):
            raise RuntimeError(f"could not set target object {resolved_target!r}")

        start_obj = arm.last_obj_pos
        start_eef = arm.last_tcp_pos
        initial_dist = _dist(start_eef, start_obj)
        if initial_dist is None:
            raise RuntimeError("failed to read initial EEF/object positions")
        if start_obj is None or len(start_obj) < 3:
            raise RuntimeError("failed to read initial object position")

        initial_obj_z = float(start_obj[2])
        best_dist = initial_dist
        prev_dist = initial_dist

        chunk_limit = max(1, int(cfg.chunk_steps or 1))
        for _outer_idx in range(max_steps):
            if getattr(arm, "episode_done", False):
                if _env_task_success(arm):
                    task_success = True
                    success_step = steps_applied
                break

            image_bytes = camera.capture_image()
            wrist_image_bytes = None
            capture_wrist = getattr(camera, "capture_wrist_image", None)
            if callable(capture_wrist):
                wrist_image_bytes = capture_wrist()
            arm_state = getattr(arm, "agent_state", None)
            state_list = arm_state.tolist() if arm_state is not None else None

            try:
                chunk = client.predict(
                    image_bytes,
                    task_prompt,
                    state=state_list,
                    wrist_image_bytes=wrist_image_bytes,
                )
            except VLAServiceError as exc:
                error = str(exc)
                break

            for raw_action in chunk[:chunk_limit]:
                if steps_applied >= max_steps:
                    break
                if getattr(arm, "episode_done", False):
                    if _env_task_success(arm):
                        task_success = True
                        if success_step is None:
                            success_step = steps_applied
                    break

                grip_value = _apply_action(arm, cfg, list(raw_action))
                if grip_value <= -abs(gripper_close_threshold):
                    close_steps += 1

                eef = arm.last_tcp_pos
                obj = arm.last_obj_pos
                dist = _dist(eef, obj)

                dist_delta = None
                if dist is not None:
                    if prev_dist is not None and dist < prev_dist - _DIST_EPS:
                        improving_steps += 1
                    if prev_dist is not None:
                        dist_delta = dist - prev_dist
                    prev_dist = dist
                    if best_dist is None or dist < best_dist:
                        best_dist = dist

                if obj is not None and len(obj) >= 3:
                    lift = float(obj[2]) - initial_obj_z
                    if lift > max_lift:
                        max_lift = lift
                    if lift_step is None and lift >= min_lift_delta:
                        lift_step = steps_applied + 1
                        if success_step is None:
                            success_step = lift_step

                if _env_task_success(arm):
                    task_success = True
                    if success_step is None:
                        success_step = steps_applied + 1

                steps_applied += 1
                trace.append(
                    {
                        "step": steps_applied,
                        "dist": None if dist is None else round(dist, 4),
                        "delta": None if dist_delta is None else round(dist_delta, 4),
                        "best_dist": None if best_dist is None else round(best_dist, 4),
                        "raw_grip": round(float(grip_value), 3),
                        "lift": round(max_lift, 4),
                    }
                )

                if task_success:
                    break

            if task_success:
                break

    except Exception as exc:  # pragma: no cover - exercised in live runs
        error = str(exc)
        logger.error("M8 probe execution error: %s", exc)
    finally:
        _close_env(env)

    return {
        "task_prompt": task_prompt,
        "target_body": resolved_target,
        "seed": seed,
        "initial_dist": initial_dist,
        "best_dist": best_dist,
        "distance_drop": (
            None
            if initial_dist is None or best_dist is None
            else round(initial_dist - best_dist, 4)
        ),
        "improving_steps": improving_steps,
        "close_steps": close_steps,
        "max_lift": round(max_lift, 4),
        "lift_step": lift_step,
        "task_success": task_success,
        "success_step": success_step,
        "steps_applied": steps_applied,
        "trace": trace,
        "error": error,
        "completed_without_error": error is None,
    }


def evaluate_m8(
    *,
    result: dict[str, Any],
    approach_dist_threshold: float,
    min_dist_drop: float,
    min_improving_steps: int,
    min_lift_delta: float,
    max_success_steps: int,
) -> tuple[str, str]:
    """Return (status, reason) for M8 policy-health acceptance."""
    if not result.get("completed_without_error", False):
        return "fail", result.get("error", "probe raised an exception")

    initial_dist = result.get("initial_dist")
    best_dist = result.get("best_dist")
    if initial_dist is None or best_dist is None:
        return "fail", "missing initial/best distance metrics"

    dist_drop = float(initial_dist) - float(best_dist)
    improving_steps = int(result.get("improving_steps", 0))
    approach_ok = (
        best_dist <= approach_dist_threshold
        and dist_drop >= min_dist_drop
        and improving_steps >= min_improving_steps
    )
    if not approach_ok:
        return (
            "fail",
            "approach not decisive "
            f"(best_dist={best_dist:.3f}m, dist_drop={dist_drop:.3f}m, "
            f"improving_steps={improving_steps})",
        )

    close_steps = int(result.get("close_steps", 0))
    if close_steps <= 0:
        return "fail", "no gripper-close commands detected"

    max_lift = float(result.get("max_lift", 0.0))
    task_success = bool(result.get("task_success", False))
    lift_confirmed = max_lift >= min_lift_delta
    if not (task_success or lift_confirmed):
        return (
            "fail",
            f"no success/lift confirmation (task_success={task_success}, max_lift={max_lift:.3f}m)",
        )

    success_step = result.get("success_step")
    if success_step is None:
        success_step = int(result.get("steps_applied", 0))
    if int(success_step) > max_success_steps:
        return (
            "fail",
            f"success/lift happened too late (success_step={success_step}, max_allowed={max_success_steps})",
        )

    return (
        "pass",
        "policy health confirmed "
        f"(best_dist={best_dist:.3f}m, dist_drop={dist_drop:.3f}m, "
        f"lift={max_lift:.3f}m, success_step={success_step})",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M8 isolated SmolVLA policy-health probe."
    )
    parser.add_argument("--task-suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-idx", type=int, default=DEFAULT_TASK_IDX)
    parser.add_argument(
        "--task-prompt",
        default=os.environ.get("M8_PROBE_TASK", DEFAULT_TASK_PROMPT),
    )
    parser.add_argument(
        "--target-body",
        default=os.environ.get("M8_TARGET_BODY", "").strip() or None,
    )
    parser.add_argument(
        "--vla-url",
        default=os.environ.get(
            "VLA_URL",
            f"http://localhost:{os.environ.get('VLA_PORT', '8765')}",
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--max-success-steps",
        type=int,
        default=int(os.environ.get("M8_MAX_SUCCESS_STEPS", DEFAULT_MAX_STEPS)),
    )
    parser.add_argument(
        "--approach-dist-threshold",
        type=float,
        default=float(
            os.environ.get("M8_APPROACH_DIST_THRESHOLD", DEFAULT_APPROACH_DIST)
        ),
    )
    parser.add_argument(
        "--min-dist-drop",
        type=float,
        default=float(os.environ.get("M8_MIN_DIST_DROP", DEFAULT_MIN_DIST_DROP)),
    )
    parser.add_argument(
        "--min-improving-steps",
        type=int,
        default=int(
            os.environ.get("M8_MIN_IMPROVING_STEPS", DEFAULT_MIN_IMPROVING_STEPS)
        ),
    )
    parser.add_argument(
        "--min-lift-delta",
        type=float,
        default=float(os.environ.get("M8_MIN_LIFT_DELTA", DEFAULT_MIN_LIFT_DELTA)),
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=float(
            os.environ.get(
                "M8_GRIPPER_CLOSE_THRESHOLD", DEFAULT_GRIPPER_CLOSE_THRESHOLD
            )
        ),
    )
    return parser.parse_args(argv)


def _print_final_result(status: str, reason: str, result: dict[str, Any]) -> None:
    label = "PASS" if status == "pass" else "FAIL"
    print(f"M8 probe {label}: {reason}")
    print(
        "initial_dist_m={initial_dist} best_dist_m={best_dist} "
        "distance_drop_m={distance_drop} improving_steps={improving_steps} "
        "close_steps={close_steps} max_lift_m={max_lift} "
        "task_success={task_success} success_step={success_step} "
        "steps_applied={steps_applied} target_body={target_body} "
        "task_prompt={task_prompt!r}".format(**result)
    )
    trace = result.get("trace") or []
    print("distance_trace_m=" + json.dumps([row.get("dist") for row in trace]))
    print("step_trace=" + json.dumps(trace, separators=(",", ":")))


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
    result: dict[str, Any] = {}
    try:
        result = run_m8_policy_health_probe(
            task_suite=args.task_suite,
            task_idx=args.task_idx,
            task_prompt=args.task_prompt,
            target_body=args.target_body,
            server_url=args.vla_url,
            seed=seed,
            max_steps=max(1, args.max_steps),
            min_lift_delta=max(0.0, args.min_lift_delta),
            gripper_close_threshold=max(0.0, args.gripper_close_threshold),
        )
        status, reason = evaluate_m8(
            result=result,
            approach_dist_threshold=max(0.0, args.approach_dist_threshold),
            min_dist_drop=max(0.0, args.min_dist_drop),
            min_improving_steps=max(0, args.min_improving_steps),
            min_lift_delta=max(0.0, args.min_lift_delta),
            max_success_steps=max(1, args.max_success_steps),
        )
    except Exception:  # pragma: no cover - top-level safety net
        reason = traceback.format_exc(limit=12)
        logger.error("M8 probe failed:\n%s", reason)
        result = {
            "initial_dist": None,
            "best_dist": None,
            "distance_drop": None,
            "improving_steps": 0,
            "close_steps": 0,
            "max_lift": 0.0,
            "task_success": False,
            "success_step": None,
            "steps_applied": 0,
            "target_body": args.target_body,
            "task_prompt": args.task_prompt,
            "trace": [],
        }

    _print_final_result(status, reason, result)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
