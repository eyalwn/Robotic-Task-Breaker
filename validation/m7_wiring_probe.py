"""M7.2/M7.3 isolated VLA→arm wiring probe (no planner/executor/LLM)."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import find_dotenv, load_dotenv

DEFAULT_GATE = "m7.2"
SUPPORTED_GATES = ("m7.2", "m7.3")
REAL_PERCEPTION_SKILLS = (
    "state.list_objects",
    "perception.detect",
    "perception.locate",
)
DEFAULT_SUITE = "libero_object"
DEFAULT_TASK_IDX = 0
DEFAULT_STEPS = 8
DEFAULT_MIN_DISPLACEMENT = 0.001
DEFAULT_TASK_PROMPT = "pick up the object"

logger = logging.getLogger(__name__)


class SimGtFallbackError(RuntimeError):
    """Raised when sim-GT perception registration is attempted in real mode."""


def validate_gate(gate: str) -> None:
    """Reject unsupported M7 gate ids."""
    if gate not in SUPPORTED_GATES:
        supported = ", ".join(SUPPORTED_GATES)
        raise ValueError(f"Unsupported gate {gate!r}; supported: {supported}")


def require_real_perception_backend() -> str:
    """Fail fast unless PERCEPTION_BACKEND resolves to real."""
    from sim.libero_perception import resolve_perception_backend

    mode = resolve_perception_backend()
    if mode != "real":
        raise ValueError(
            f"M7.3 requires PERCEPTION_BACKEND=real, got {mode!r}"
        )
    return mode


@contextmanager
def sim_gt_fallback_guard() -> Iterator[None]:
    """Block sim-GT perception registration for the guarded scope."""
    import sim.libero_skills as libero_skills_mod

    original = libero_skills_mod.register_libero_skills

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise SimGtFallbackError(
            "sim-GT perception registration attempted while PERCEPTION_BACKEND=real"
        )

    libero_skills_mod.register_libero_skills = _blocked
    try:
        yield
    finally:
        libero_skills_mod.register_libero_skills = original


def verify_real_perception_wiring(*, vec_env: Any, camera: Any) -> dict[str, Any]:
    """Register perception skills and assert the real path is used."""
    from core.skills import SkillRegistry
    from sim.libero_perception import register_libero_perception_skills

    registry = SkillRegistry()
    mode = register_libero_perception_skills(
        registry,
        vec_env=vec_env,
        camera=camera,
        mode="real",
    )
    sources = {
        name: getattr(registry.get(name), "perception_source", None)
        for name in REAL_PERCEPTION_SKILLS
    }
    if mode != "real":
        raise RuntimeError(
            f"expected real perception registration, got mode={mode!r}"
        )
    bad = {name: source for name, source in sources.items() if source != "real"}
    if bad:
        raise RuntimeError(
            "expected real perception skills, got non-real sources: "
            + ", ".join(f"{name}={source!r}" for name, source in bad.items())
        )
    return {
        "perception_mode": mode,
        "perception_sources": sources,
        "verified": True,
    }


def eef_displacement(
    before: list[float] | None,
    after: list[float] | None,
) -> float | None:
    """Return L2 norm of EE motion, or None when either pose is missing."""
    if before is None or after is None:
        return None
    if len(before) < 3 or len(after) < 3:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(before[:3], after[:3])))


def evaluate_m7_2(
    *,
    completed_without_error: bool,
    displacement: float | None,
    min_displacement: float,
) -> tuple[str, str]:
    """Return (status, reason) for the M7.2 gate."""
    if not completed_without_error:
        return "fail", "probe raised an exception"
    if displacement is None:
        return "fail", "missing EE position before/after probe"
    if displacement < min_displacement:
        return (
            "fail",
            f"EE displacement {displacement:.6f}m below threshold {min_displacement:.6f}m",
        )
    return "pass", f"EE displaced {displacement:.6f}m (>= {min_displacement:.6f}m)"


def evaluate_m7_3(
    *,
    completed_without_error: bool,
    displacement: float | None,
    min_displacement: float,
    perception_verified: bool,
    perception_reason: str | None = None,
) -> tuple[str, str]:
    """Return (status, reason) for the M7.3 gate."""
    if not perception_verified:
        return "fail", perception_reason or "real perception registration not verified"
    return evaluate_m7_2(
        completed_without_error=completed_without_error,
        displacement=displacement,
        min_displacement=min_displacement,
    )


def ensure_libero_config() -> str:
    """Create ~/.libero/config.yaml when missing (headless-safe)."""
    config_root = Path(
        os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))
    )
    config_path = config_root / "config.yaml"
    if config_path.exists():
        return str(config_path)

    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise RuntimeError("libero package not found while preparing LIBERO config")

    libero_pkg = Path(spec.origin).resolve().parent / "libero"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"benchmark_root: {libero_pkg.as_posix()}",
                f"bddl_files: {(libero_pkg / 'bddl_files').as_posix()}",
                f"init_states: {(libero_pkg / 'init_files').as_posix()}",
                f"datasets: {(libero_pkg.parent / 'datasets').as_posix()}",
                f"assets: {(libero_pkg / 'assets').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Created LIBERO config file at %s", config_path)
    return str(config_path)


def _apply_action(arm: Any, cfg: Any, raw_action: list[float]) -> None:
    from vla.action import process_raw_action

    if len(raw_action) == 7:
        clipped = [max(-cfg.action_clip, min(cfg.action_clip, v)) for v in raw_action]
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
        return
    if len(raw_action) == 4:
        dx, dy, dz = process_raw_action(raw_action[:3], cfg)
        arm.move_ee(dx, dy, dz, steps=cfg.steps_per_action)
        return
    raise ValueError(f"VLA action must have 4 or 7 elements, got {len(raw_action)}")


def _close_env(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.warning("env.close() failed: %s", exc)


def run_wiring_probe(
    *,
    task_suite: str,
    task_idx: int,
    steps: int,
    task_prompt: str,
    server_url: str,
    seed: int | None,
    gate: str = DEFAULT_GATE,
) -> dict[str, Any]:
    """Run one bounded predict/apply loop and return metrics."""
    from sim.libero_backend import LiberoArm, LiberoCamera, create_env
    from sim.libero_smoke import apply_smoke_env_defaults
    from vla.client import VLAClient, VLAServiceError
    from vla.config import VLAConfig

    apply_smoke_env_defaults()
    if seed is not None:
        os.environ["ROBOCREW_SIM_SEED"] = str(seed)
    ensure_libero_config()

    cfg = VLAConfig.from_env()
    client = VLAClient(server_url, timeout=cfg.request_timeout)
    client.health_check()

    perception_metrics: dict[str, Any] | None = None
    perception_error: str | None = None

    with sim_gt_fallback_guard() if gate == "m7.3" else _noop_context():
        env = create_env(task_suite=task_suite, task_idx=task_idx)
        arm = LiberoArm(env)
        camera = LiberoCamera(env, arm=arm)

        if gate == "m7.3":
            try:
                perception_metrics = verify_real_perception_wiring(
                    vec_env=env,
                    camera=camera,
                )
            except (SimGtFallbackError, RuntimeError, ValueError) as exc:
                perception_error = str(exc)
                logger.error("M7.3 perception verification failed: %s", exc)
                _close_env(env)
                return {
                    "ee_before": None,
                    "ee_after": None,
                    "displacement_m": None,
                    "steps_applied": 0,
                    "seed": seed,
                    "task_prompt": task_prompt,
                    "error": perception_error,
                    "completed_without_error": False,
                    "perception": {
                        "verified": False,
                        "error": perception_error,
                    },
                }

        ee_before = list(arm.last_tcp_pos) if arm.last_tcp_pos else None
        steps_applied = 0
        last_error: str | None = None

        try:
            chunk_limit = max(1, int(cfg.chunk_steps or 1))
            while steps_applied < steps:
                if getattr(arm, "episode_done", False):
                    logger.warning("Episode ended early at step %d", steps_applied)
                    break

                image_bytes = camera.capture_image()
                wrist_image_bytes = None
                capture_wrist = getattr(camera, "capture_wrist_image", None)
                if callable(capture_wrist):
                    wrist_image_bytes = capture_wrist()

                arm_state = getattr(arm, "agent_state", None)
                state_list = arm_state.tolist() if arm_state is not None else None
                chunk = client.predict(
                    image_bytes,
                    task_prompt,
                    state=state_list,
                    wrist_image_bytes=wrist_image_bytes,
                )

                for raw_action in chunk[:chunk_limit]:
                    if steps_applied >= steps:
                        break
                    if getattr(arm, "episode_done", False):
                        break
                    _apply_action(arm, cfg, list(raw_action))
                    steps_applied += 1
        except SimGtFallbackError as exc:
            last_error = str(exc)
            logger.error("M7.3 sim-GT fallback guard tripped: %s", exc)
        except VLAServiceError as exc:
            last_error = str(exc)
            logger.error("VLA service error: %s", exc)
        finally:
            ee_after = list(arm.last_tcp_pos) if arm.last_tcp_pos else None
            _close_env(env)

    displacement = eef_displacement(ee_before, ee_after)
    result = {
        "ee_before": ee_before,
        "ee_after": ee_after,
        "displacement_m": displacement,
        "steps_applied": steps_applied,
        "seed": seed,
        "task_prompt": task_prompt,
        "error": last_error,
        "completed_without_error": last_error is None,
    }
    if gate == "m7.3":
        result["perception"] = perception_metrics or {
            "verified": False,
            "error": perception_error,
        }
    return result


@contextmanager
def _noop_context() -> Iterator[None]:
    yield


def _drift_between_runs(first: dict[str, Any], second: dict[str, Any]) -> float | None:
    return eef_displacement(first.get("ee_after"), second.get("ee_after"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M7 wiring probe (isolated VLA→arm).")
    parser.add_argument(
        "--gate",
        default=DEFAULT_GATE,
        help="M7 gate id (default: m7.2; m7.3 requires PERCEPTION_BACKEND=real)",
    )
    parser.add_argument("--task-suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-idx", type=int, default=DEFAULT_TASK_IDX)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=DEFAULT_MIN_DISPLACEMENT,
        help="Minimum EE displacement (m) required to pass",
    )
    parser.add_argument(
        "--task-prompt",
        default=os.environ.get("M7_PROBE_TASK", DEFAULT_TASK_PROMPT),
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
    parser.add_argument(
        "--drift-check",
        action="store_true",
        help="Run two short probes with the same seed and report EE drift",
    )
    return parser.parse_args(argv)


def _print_final_result(
    *,
    gate: str,
    status: str,
    reason: str,
    primary: dict[str, Any] | None,
    drift_m: float | None,
    min_displacement: float,
) -> None:
    label = "PASS" if status == "pass" else "FAIL"
    print(f"{gate.upper()} probe {label}: {reason}")
    if primary is not None:
        print(
            f"displacement_m={primary.get('displacement_m')} "
            f"steps_applied={primary.get('steps_applied')} "
            f"min_displacement_m={min_displacement}"
        )
        if gate == "m7.3":
            perception = primary.get("perception") or {}
            print(f"perception_verified={perception.get('verified', False)}")
    if drift_m is not None:
        print(f"drift_m={drift_m}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(find_dotenv())
    args = parse_args(argv)

    try:
        validate_gate(args.gate)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from sim.libero_smoke import configure_smoke_logging

    configure_smoke_logging()

    seed = args.seed
    if seed is None:
        raw_seed = os.environ.get("ROBOCREW_SIM_SEED", "").strip()
        seed = int(raw_seed) if raw_seed else 0

    status = "fail"
    reason = "not started"
    primary: dict[str, Any] | None = None
    drift_m: float | None = None

    try:
        if args.gate == "m7.3":
            require_real_perception_backend()

        logger.info(
            "%s probe start: suite=%s task_idx=%s steps=%s seed=%s",
            args.gate.upper(),
            args.task_suite,
            args.task_idx,
            args.steps,
            seed,
        )
        primary = run_wiring_probe(
            task_suite=args.task_suite,
            task_idx=args.task_idx,
            steps=args.steps,
            task_prompt=args.task_prompt,
            server_url=args.vla_url,
            seed=seed,
            gate=args.gate,
        )

        if args.drift_check:
            repeat_steps = max(3, min(args.steps, 5))
            repeat = run_wiring_probe(
                task_suite=args.task_suite,
                task_idx=args.task_idx,
                steps=repeat_steps,
                task_prompt=args.task_prompt,
                server_url=args.vla_url,
                seed=seed,
                gate=args.gate,
            )
            drift_m = _drift_between_runs(primary, repeat)

        if args.gate == "m7.3":
            perception = primary.get("perception") or {}
            status, reason = evaluate_m7_3(
                completed_without_error=primary.get("completed_without_error", False),
                displacement=primary.get("displacement_m"),
                min_displacement=args.min_displacement,
                perception_verified=bool(perception.get("verified")),
                perception_reason=perception.get("error"),
            )
        else:
            status, reason = evaluate_m7_2(
                completed_without_error=primary.get("completed_without_error", False),
                displacement=primary.get("displacement_m"),
                min_displacement=args.min_displacement,
            )
    except ValueError as exc:
        reason = str(exc)
        logger.error("%s probe config error: %s", args.gate.upper(), reason)
    except Exception:
        reason = traceback.format_exc(limit=12)
        logger.error("%s probe failed:\n%s", args.gate.upper(), reason)

    _print_final_result(
        gate=args.gate,
        status=status,
        reason=reason,
        primary=primary,
        drift_m=drift_m,
        min_displacement=args.min_displacement,
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
