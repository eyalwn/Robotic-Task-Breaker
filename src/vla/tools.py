from __future__ import annotations

import itertools
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import tool

from .action import process_raw_action, requested_gripper_state, should_flip_gripper
from .client import VLAClient, VLAServiceError
from .config import VLAConfig


logger = logging.getLogger(__name__)

_obs_keys_dumped = False
_VLA_DEBUG_TELEMETRY_MAX_STEPS = 20


def _vla_debug_action_telemetry_enabled() -> bool:
    raw = os.environ.get("VLA_DEBUG_ACTION_TELEMETRY", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _preview_osc_action(
    arm: Any,
    dx: float,
    dy: float,
    dz: float,
    drx: float = 0.0,
    dry: float = 0.0,
    drz: float = 0.0,
    grip: float | None = None,
) -> list[float]:
    """Mirror LiberoArm OSC clip logic when last_osc_action is unavailable."""
    import numpy as np

    sent = getattr(arm, "last_osc_action", None)
    if sent is not None:
        flat = np.asarray(sent, dtype=np.float32).reshape(-1)
        if flat.size >= 7:
            return flat[:7].tolist()

    pos_scale = float(getattr(arm, "_pos_scale", 1.0))
    deltas = np.clip(
        pos_scale * np.array([dx, dy, dz, drx, dry, drz], dtype=np.float32),
        -1.0,
        1.0,
    )
    grip_val = float(grip if grip is not None else getattr(arm, "_gripper_value", 0.0))
    return [*deltas.tolist(), float(np.clip(grip_val, -1.0, 1.0))]


def _log_action_telemetry(
    step: int,
    raw_action: list[float],
    scaled: tuple[float, float, float],
    rot_grip: tuple[float, float, float, float] | None,
    arm: Any,
    state: "_RunState",
    eef: list[float] | None,
    dist: float | None,
) -> None:
    if not _vla_debug_action_telemetry_enabled() or step >= _VLA_DEBUG_TELEMETRY_MAX_STEPS:
        return

    if step == 0:
        agent_state = getattr(arm, "agent_state", None)
        if agent_state is not None:
            import numpy as np

            flat = np.asarray(agent_state, dtype=np.float32).reshape(-1)[:8]
            logger.info(
                "VLA telem step=00 agent_state=%s",
                [f"{v:+.4f}" for v in flat.tolist()],
            )

    dx, dy, dz = scaled
    if rot_grip is not None:
        drx, dry, drz, grip = rot_grip
    else:
        drx = dry = drz = 0.0
        grip = raw_action[3] if len(raw_action) >= 4 else 0.0

    osc = _preview_osc_action(arm, dx, dy, dz, drx, dry, drz, grip)
    eef_delta = "?"
    if state.prev_eef and eef:
        eef_delta = (
            f"[{eef[0] - state.prev_eef[0]:+.4f},"
            f"{eef[1] - state.prev_eef[1]:+.4f},"
            f"{eef[2] - state.prev_eef[2]:+.4f}]"
        )
    min_dist_str = f"{state.min_dist:.4f}" if state.min_dist is not None else "?"

    logger.info(
        "VLA telem step=%02d raw=%s scaled=[%+.4f,%+.4f,%+.4f] "
        "rot=[%+.4f,%+.4f,%+.4f] grip=%+.4f osc=%s eef_d=%s min_dist=%s",
        step,
        [f"{v:+.4f}" for v in raw_action],
        dx,
        dy,
        dz,
        drx,
        dry,
        drz,
        grip,
        [f"{v:+.4f}" for v in osc[:6]],
        eef_delta,
        min_dist_str,
    )


def _round_floats(obj: Any, ndigits: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def _dump_obs_keys(obs, prefix: str = "") -> None:
    """Recursively log all keys in a nested dict/mapping (one-time diagnostic)."""
    if not isinstance(obs, dict):
        import numpy as np

        try:
            array = np.asarray(obs)
            logger.debug(
                "  %s → shape=%s dtype=%s",
                prefix or "(root)",
                array.shape,
                array.dtype,
            )
        except Exception:
            logger.debug(
                "  %s → type=%s",
                prefix or "(root)",
                type(obs).__name__,
            )
        return
    for key, value in obs.items():
        _dump_obs_keys(value, f"{prefix}.{key}" if prefix else key)


def _extract_positions(obs) -> tuple[list[float] | None, list[float] | None]:
    """Best-effort extraction of EEF and object xyz from an observation mapping.

    Returns (eef_xyz, obj_xyz) — each a 3-element list or None.
    On first call, dumps the full obs key structure so missing paths can be diagnosed.
    """
    global _obs_keys_dumped
    if obs is None or not isinstance(obs, dict):
        return None, None

    if not _obs_keys_dumped:
        logger.debug("Obs dict structure (one-time dump):")
        _dump_obs_keys(obs)
        _obs_keys_dumped = True

    import numpy as np

    def _to_list(value):
        array = np.asarray(value).flatten()
        return array[:3].tolist() if array.size >= 3 else None

    def _walk(mapping, path):
        value = mapping
        for key in path:
            try:
                value = value[key]
            except (KeyError, TypeError, IndexError):
                return None
        return _to_list(value)

    eef = None
    for candidate in (
        ("extra", "tcp_pose"),
        ("agent", "tcp_pose"),
        ("extra", "ee_pos"),
    ):
        eef = _walk(obs, candidate)
        if eef is not None:
            break

    obj = None
    for candidate in (
        ("extra", "obj_pose"),
        ("extra", "goal_pos"),
    ):
        obj = _walk(obs, candidate)
        if obj is not None:
            break

    return eef, obj


class _CameraLike(Protocol):
    def capture_image(self) -> bytes: ...


class _ArmLike(Protocol):
    last_tcp_pos: list[float] | None
    last_obj_pos: list[float] | None
    last_obs: Any

    def move_ee(self, dx: float, dy: float, dz: float, steps: int = 1) -> Any: ...

    def set_gripper(self, state: str, steps: int = 1) -> Any: ...


def _clip_action(value: float, cfg: VLAConfig) -> float:
    return max(-cfg.action_clip, min(cfg.action_clip, value))


def _use_raw_openpi_action_path(cfg: VLAConfig) -> bool:
    """OpenPI LIBERO eval: no axis remap/signs and no adaptive scaling."""
    return (
        not cfg.adaptive_scale
        and cfg.axis_remap is None
        and cfg.axis_signs is None
    )


def _proximity_grasp_enabled(cfg: VLAConfig) -> bool:
    """Proximity early-exit is for ManiSkill/PickCube; disabled on OpenPI LIBERO."""
    if not cfg.proximity_grasp:
        return False
    if _use_raw_openpi_action_path(cfg):
        return False
    return True


def _apply_raw_action_to_arm(
    arm: _ArmLike | Any,
    cfg: VLAConfig,
    state: _RunState,
    chunk: _ChunkState,
    raw_action: list[float],
) -> tuple[float, float, float, float, list[float], tuple[float, float, float, float] | None]:
    """Apply one raw VLA action; returns (dx, dy, dz, sf, raw_xyz, rot_grip_or_none)."""
    if len(raw_action) == 7:
        raw_dx, raw_dy, raw_dz, raw_drx, raw_dry, raw_drz, raw_grip = raw_action
        raw_xyz = [raw_dx, raw_dy, raw_dz]
        apply_policy = getattr(arm, "apply_policy_action", None)
        apply_vla = getattr(arm, "apply_vla_action", None)

        if not callable(apply_vla) and not callable(apply_policy):
            raise RuntimeError(
                "VLA returned 7-dim actions but arm has no apply_policy_action() "
                "or apply_vla_action()"
            )

        if _use_raw_openpi_action_path(cfg):
            if callable(apply_policy):
                clipped = [_clip_action(v, cfg) for v in raw_action]
                apply_policy(clipped, steps=cfg.steps_per_action)
                return (
                    clipped[0],
                    clipped[1],
                    clipped[2],
                    1.0,
                    clipped[:3],
                    tuple(clipped[3:7]),
                )
            dx, dy, dz = [_clip_action(v, cfg) for v in raw_xyz]
            drx = _clip_action(raw_drx, cfg)
            dry = _clip_action(raw_dry, cfg)
            drz = _clip_action(raw_drz, cfg)
            grip = _clip_action(raw_grip, cfg)
            apply_vla(
                dx,
                dy,
                dz,
                drx,
                dry,
                drz,
                grip,
                steps=cfg.steps_per_action,
            )
            return dx, dy, dz, 1.0, raw_xyz, (drx, dry, drz, grip)

        _maybe_autocalibrate_axes(state, cfg, raw_xyz)
        dx, dy, dz = process_raw_action(
            raw_xyz,
            cfg,
            axis_remap=state.active_axis_remap,
            axis_signs=state.active_axis_signs,
        )
        dx, dy, dz, sf = _apply_adaptive_scaling(dx, dy, dz, cfg, state.pre_dist)
        action_vec = [dx, dy, dz, raw_drx, raw_dry, raw_drz, raw_grip]
        if callable(apply_policy):
            apply_policy(action_vec, steps=cfg.steps_per_action)
        else:
            apply_vla(
                dx,
                dy,
                dz,
                raw_drx,
                raw_dry,
                raw_drz,
                raw_grip,
                steps=cfg.steps_per_action,
            )
        return dx, dy, dz, sf, raw_xyz, (raw_drx, raw_dry, raw_drz, raw_grip)

    if len(raw_action) != 4:
        raise ValueError(
            f"VLA action must have 4 or 7 elements, got {len(raw_action)}"
        )

    raw_dx, raw_dy, raw_dz, raw_gripper = raw_action
    raw_xyz = [raw_dx, raw_dy, raw_dz]
    _maybe_autocalibrate_axes(state, cfg, raw_xyz)
    dx, dy, dz = process_raw_action(
        raw_xyz,
        cfg,
        axis_remap=state.active_axis_remap,
        axis_signs=state.active_axis_signs,
    )
    dx, dy, dz, sf = _apply_adaptive_scaling(dx, dy, dz, cfg, state.pre_dist)
    arm.move_ee(dx, dy, dz, steps=cfg.steps_per_action)
    eef, obj, dist = _eef_obj_dist(arm)
    _maybe_apply_gripper(arm, cfg, state, chunk, raw_gripper, dist)
    return dx, dy, dz, sf, raw_xyz, None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _eef_obj_dist(arm: Any) -> tuple[list[float] | None, list[float] | None, float | None]:
    eef = getattr(arm, "last_tcp_pos", None)
    obj = getattr(arm, "last_obj_pos", None)
    if eef is None and obj is None:
        eef, obj = _extract_positions(getattr(arm, "last_obs", None))
    dist: float | None = None
    if eef and obj:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(eef, obj)))
    return eef, obj, dist


def _axis_calibration_hint(
    eef: list[float] | None,
    obj: list[float] | None,
    raw_xyz: list[float],
) -> tuple[float, tuple[int, int, int] | None, tuple[int, int, int] | None]:
    if not eef or not obj:
        return -1.0, None, None
    if not any(v != 0.0 for v in raw_xyz):
        return -1.0, None, None

    ideal = [b - a for a, b in zip(eef, obj)]
    ideal_norm = math.sqrt(sum(v**2 for v in ideal))
    raw_norm = math.sqrt(sum(v**2 for v in raw_xyz))
    if ideal_norm <= 0.0 or raw_norm <= 0.0:
        return -1.0, None, None

    ideal_unit = [v / ideal_norm for v in ideal]
    raw_unit = [v / raw_norm for v in raw_xyz]

    best_cos = -1.0
    best_perm = None
    best_signs = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            mapped = [signs[i] * raw_unit[perm[i]] for i in range(3)]
            cosine = sum(a * b for a, b in zip(mapped, ideal_unit))
            if cosine > best_cos:
                best_cos = cosine
                best_perm = perm
                best_signs = signs
    return best_cos, best_perm, best_signs


@dataclass
class _RunState:
    steps_applied: int = 0
    last_raw_action: list[float] | None = None
    min_dist: float | None = None
    best_min_dist: float | None = None
    stagnation_count: int = 0
    prev_eef: list[float] | None = None
    gripper_state: str = "open"
    pre_eef: list[float] | None = None
    pre_obj: list[float] | None = None
    pre_dist: float | None = None
    active_axis_remap: list[int] | None = None
    active_axis_signs: list[int] | None = None
    last_ik_warn_step: int = -10_000
    fail_dist_count: int = 0
    worsen_count: int = 0
    ik_saturation_count: int = 0
    initial_obj_z: float | None = None


@dataclass
class _ChunkState:
    suppressed_gripper_log: bool = False
    max_raw_mag: float = 0.0
    ik_saturation: bool = False


def _maybe_autocalibrate_axes(
    state: _RunState,
    cfg: VLAConfig,
    raw_xyz: list[float],
) -> None:
    if state.steps_applied != 0:
        return
    best_cos, best_perm, best_signs = _axis_calibration_hint(
        state.pre_eef,
        state.pre_obj,
        raw_xyz,
    )
    if best_perm is None or best_signs is None:
        return

    new_remap = list(best_perm)
    new_signs = [float(s) for s in best_signs]

    logger.debug(
        "AXIS CALIBRATION HINT: ideal_dir from eef→obj "
        "| raw_dir=%s | best_remap=%s best_signs=%s cosine=%.4f | "
        "cfg_remap=%s cfg_signs=%s active_remap=%s active_signs=%s",
        [f"{v:.4f}" for v in raw_xyz],
        new_remap,
        new_signs,
        best_cos,
        cfg.axis_remap,
        cfg.axis_signs,
        state.active_axis_remap,
        state.active_axis_signs,
    )

    if not cfg.auto_calibrate:
        logger.debug(
            "VLA axis calibration: to match hint set "
            "VLA_AXIS_REMAP=%s VLA_AXIS_SIGNS=%s (cosine=%.4f)",
            ",".join(str(i) for i in new_remap),
            ",".join(str(int(s)) if s == int(s) else str(s) for s in new_signs),
            best_cos,
        )
        return

    if best_cos < cfg.auto_calibrate_min_cosine:
        logger.debug(
            "AXIS CALIBRATION SKIPPED: best cosine %.4f below threshold %.4f",
            best_cos,
            cfg.auto_calibrate_min_cosine,
        )
        return

    prev_remap = (
        list(state.active_axis_remap) if state.active_axis_remap is not None else None
    )
    prev_signs = (
        [float(s) for s in state.active_axis_signs]
        if state.active_axis_signs is not None
        else None
    )

    state.active_axis_remap = new_remap
    state.active_axis_signs = new_signs

    if prev_remap != new_remap or prev_signs != new_signs:
        logger.info(
            "VLA axis calibration applied: VLA_AXIS_REMAP=%s VLA_AXIS_SIGNS=%s "
            "cosine=%.4f (threshold=%.4f)",
            ",".join(str(i) for i in new_remap),
            ",".join(str(int(s)) if s == int(s) else str(s) for s in new_signs),
            best_cos,
            cfg.auto_calibrate_min_cosine,
        )


def _apply_adaptive_scaling(
    dx: float,
    dy: float,
    dz: float,
    cfg: VLAConfig,
    pre_dist: float | None,
) -> tuple[float, float, float, float]:
    sf = 1.0
    if cfg.adaptive_scale and pre_dist is not None and cfg.adaptive_ref_dist > 0.0:
        sf = _clamp(
            pre_dist / cfg.adaptive_ref_dist,
            cfg.adaptive_min_factor,
            1.0,
        )
        dx *= sf
        dy *= sf
        dz *= sf
    return dx, dy, dz, sf


def _maybe_apply_gripper(
    arm: _ArmLike | Any,
    cfg: VLAConfig,
    state: _RunState,
    chunk: _ChunkState,
    raw_gripper: float,
    dist: float | None,
) -> None:
    clipped = max(-cfg.action_clip, min(cfg.action_clip, raw_gripper))
    requested = requested_gripper_state(clipped, cfg)
    wants_flip = requested != state.gripper_state
    if (
        cfg.sim_pickcube_gripper_profile
        and requested == "close"
        and dist is not None
        and dist > cfg.gripper_close_max_dist
    ):
        logger.debug(
            "VLA step %3d | GRIPPER close suppressed: dist %.3fm > %.3fm",
            state.steps_applied,
            dist,
            cfg.gripper_close_max_dist,
        )
        return
    if should_flip_gripper(clipped, state.gripper_state, cfg):
        logger.debug(
            "VLA step %3d | GRIPPER %s (raw_gripper=%+.3f, confidence=%.3f)",
            state.steps_applied,
            requested,
            clipped,
            cfg.gripper_confidence,
        )
        arm.set_gripper(requested, steps=cfg.steps_per_action)
        state.gripper_state = requested
        return
    if wants_flip and not chunk.suppressed_gripper_log:
        logger.debug(
            "VLA step %3d | GRIPPER flip suppressed "
            "(requested=%s current=%s raw_gripper=%+.3f, confidence=%.3f)",
            state.steps_applied,
            requested,
            state.gripper_state,
            clipped,
            cfg.gripper_confidence,
        )
        chunk.suppressed_gripper_log = True


def _actual_delta_and_ik_ratio(
    state: _RunState,
    eef: list[float] | None,
    dx: float,
    dy: float,
    dz: float,
    ik_ratio_low: float,
    ik_ratio_high: float,
    ik_warn_cooldown_steps: int,
    arm: _ArmLike | Any,
) -> tuple[str, bool]:
    actual_delta_str = "?"
    chunk_ik_saturation = False
    if state.prev_eef is None or not eef:
        return actual_delta_str, chunk_ik_saturation
    actual_delta = [a - b for a, b in zip(eef, state.prev_eef)]
    actual_delta_str = [f"{value:.4f}" for value in actual_delta]
    scaled_mag = math.sqrt(dx**2 + dy**2 + dz**2)
    actual_mag = math.sqrt(sum(component**2 for component in actual_delta))
    if scaled_mag > 0.001 and actual_mag > 0.001:
        ratio = actual_mag / scaled_mag
        if ratio > ik_ratio_high or ratio < ik_ratio_low:
            chunk_ik_saturation = True
            if state.steps_applied - state.last_ik_warn_step >= ik_warn_cooldown_steps:
                ik_debug = getattr(arm, "last_ik_debug", {}) or {}
                cause = ik_debug.get("cause", "unknown")
                clipped_joints = int(ik_debug.get("clipped_joints", 0))
                jacobian_cond = ik_debug.get("jacobian_cond")
                cond_str = f"{jacobian_cond:.1e}" if isinstance(jacobian_cond, float) else "?"
                logger.warning(
                    "VLA step %d: IK tracking poor (ratio=%.2f) cause=%s "
                    "scaled=%.4fm actual=%.4fm clipped_joints=%d cond=%s",
                    state.steps_applied,
                    ratio,
                    cause,
                    scaled_mag,
                    actual_mag,
                    clipped_joints,
                    cond_str,
                )
                state.last_ik_warn_step = state.steps_applied
    return actual_delta_str, chunk_ik_saturation


def _maybe_proximity_grasp(
    arm: _ArmLike | Any,
    cfg: VLAConfig,
    state: _RunState,
    dist: float | None,
    obj: list[float] | None,
) -> tuple[str, bool] | None:
    if dist is None or dist >= cfg.grasp_dist:
        return None
    logger.info(
        "Proximity grasp: dist=%.3fm < %.3fm — closing gripper. "
        "Steps applied: %d.",
        dist,
        cfg.grasp_dist,
        state.steps_applied,
    )
    arm.set_gripper("close", steps=10)
    state.gripper_state = "close"
    if hasattr(arm, "step"):
        arm.step(steps=5)
    new_obj = getattr(arm, "last_obj_pos", None)
    lifted = False
    if new_obj and obj:
        lifted = (new_obj[2] - obj[2]) > 0.005
    logger.info(
        "Proximity grasp verification: lifted=%s (before_z=%s, after_z=%s)",
        lifted,
        obj[2] if obj else None,
        new_obj[2] if new_obj else None,
    )
    min_dist_str = f"{state.min_dist:.3f}" if state.min_dist is not None else "unknown"
    base_msg = (
        f"VLA policy: proximity grasp triggered after {state.steps_applied} steps "
        f"(dist={dist:.3f}m < {cfg.grasp_dist}m). "
        f"min_dist={min_dist_str}m. grasp_triggered=True. "
    )
    if lifted:
        result_msg = base_msg + "grasp verified (object lifted above table). "
    else:
        result_msg = (
            base_msg
            + "proximity reached but grasp not confirmed (object did not lift). "
        )
    return result_msg + "settle_steps=5, gripper_steps=10.", lifted


def _env_task_success(arm: _ArmLike | Any, cfg: VLAConfig) -> bool:
    if not cfg.use_env_task_success:
        return False
    try:
        env = getattr(arm, "_env", None)
        raw = getattr(env, "unwrapped", env)
        if raw is not None:
            info = raw.evaluate()
            if isinstance(info, dict):
                success_value = info.get("success")
                if success_value is not None:
                    try:
                        success_scalar = float(success_value)
                    except (TypeError, ValueError):
                        success_scalar = None
                    if success_scalar is not None and bool(success_scalar):
                        return True
    except Exception:
        pass

    info = getattr(arm, "last_info", {}) or {}
    for key in ("success", "is_success", "task_success"):
        if bool(info.get(key)):
            return True
    return False


def _object_lift_success(
    arm: _ArmLike | Any,
    cfg: VLAConfig,
    state: _RunState,
) -> bool:
    obj = getattr(arm, "last_obj_pos", None)
    if not obj or len(obj) < 3:
        return False
    if state.initial_obj_z is None:
        state.initial_obj_z = obj[2]
        return False
    return obj[2] - state.initial_obj_z >= cfg.success_lift_delta


def _result_payload(
    reason: str,
    state: _RunState,
    cfg: VLAConfig,
    detail: str | None = None,
    success: bool = False,
    grasp_confirmed: bool | None = None,
) -> str:
    payload = {
        "status": "success" if success else "stopped",
        "reason": reason,
        "steps_applied": state.steps_applied,
        "min_dist": round(state.min_dist, 4) if state.min_dist is not None else None,
        "best_min_dist": (
            round(state.best_min_dist, 4) if state.best_min_dist is not None else None
        ),
        "grasp_dist": cfg.grasp_dist,
        "detail": detail,
        "grasp_triggered": success,
        "last_raw_action": state.last_raw_action,
    }
    if grasp_confirmed is not None:
        payload["grasp_confirmed"] = grasp_confirmed
    payload = _round_floats(payload)
    return json.dumps(payload, separators=(",", ":"))


def _check_failure_heuristics(
    cfg: VLAConfig,
    state: _RunState,
    dist: float | None,
    chunk_ik_saturation: bool,
) -> str | None:
    if dist is not None:
        if dist > cfg.fail_dist_threshold:
            state.fail_dist_count += 1
        else:
            state.fail_dist_count = 0
        if (
            state.best_min_dist is not None
            and dist > (state.best_min_dist + cfg.fail_worsening_margin)
        ):
            state.worsen_count += 1
        else:
            state.worsen_count = 0
    if chunk_ik_saturation:
        state.ik_saturation_count += 1
    else:
        state.ik_saturation_count = 0

    if state.fail_dist_count >= cfg.fail_dist_consecutive_steps:
        return "fail_dist_consecutive"
    if state.worsen_count >= cfg.fail_worsening_steps:
        return "fail_dist_worsening"
    if state.ik_saturation_count >= cfg.fail_ik_saturation_steps:
        return "fail_ik_saturation"
    return None


def _camera_vision_unavailable(camera: _CameraLike | Any, image_bytes: bytes) -> bool:
    health = getattr(camera, "vision_health", None)
    if health is not None:
        return bool(getattr(health, "vision_unavailable", False))
    if bool(getattr(camera, "session_unusable_for_vla", False)):
        return True
    if bool(getattr(camera, "vision_degraded", False)):
        return True
    if bool(getattr(camera, "last_capture_was_placeholder", False)):
        return True
    is_placeholder = getattr(camera, "is_placeholder_jpeg", None)
    if callable(is_placeholder):
        try:
            return bool(is_placeholder(image_bytes))
        except Exception:
            return False
    return False


def _update_stagnation_after_chunk(
    arm: _ArmLike | Any,
    cfg: VLAConfig,
    state: _RunState,
    chunk: _ChunkState,
    best_min_before_chunk: float | None,
) -> str | None:
    chunk_eef = getattr(arm, "last_tcp_pos", None)
    chunk_obj = getattr(arm, "last_obj_pos", None)
    if not chunk_eef or not chunk_obj:
        return None
    chunk_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(chunk_eef, chunk_obj)))
    if state.min_dist is None or chunk_dist < state.min_dist:
        state.min_dist = chunk_dist
    min_dist_improved = (
        state.best_min_dist is not None
        and (
            best_min_before_chunk is None
            or state.best_min_dist < best_min_before_chunk
        )
    )
    if min_dist_improved:
        state.stagnation_count = 0
    else:
        state.stagnation_count += 1
        logger.debug(
            "Stagnation %d/%d (chunk_dist=%.3fm, min_dist=%.3fm, raw_mag=%.3f, ik_saturation=%s)",
            state.stagnation_count,
            cfg.stagnation_chunks,
            chunk_dist,
            state.min_dist if state.min_dist is not None else -1.0,
            chunk.max_raw_mag,
            chunk.ik_saturation,
        )
    if state.stagnation_count < cfg.stagnation_chunks:
        return None
    stop_dist = state.min_dist if state.min_dist is not None else chunk_dist
    logger.info(
        "Early stop: EEF distance stagnated for %d chunks "
        "(dist=%.3fm). Steps applied: %d.",
        state.stagnation_count,
        stop_dist,
        state.steps_applied,
    )
    min_dist_str = f"{state.min_dist:.3f}" if state.min_dist is not None else "unknown"
    return (
        f"stagnation: dist={stop_dist:.3f}m "
        f"for {state.stagnation_count} chunks, min_dist={min_dist_str}m"
    )


def create_vla_manipulation(
    tool_name: str,
    tool_description: str,
    task_prompt: str,
    server_address: str,
    arm: _ArmLike | Any,
    camera: _CameraLike | Any,
    max_steps: int = 150,
    cfg: VLAConfig | None = None,
):
    """Return a LangChain tool that runs a closed-loop VLA policy."""
    if cfg is None:
        cfg = VLAConfig()

    effective_max_steps = max(1, min(int(max_steps), int(cfg.max_inner_steps)))
    logger.debug(
        "VLA manipulation tool created — scale=%.4f axis_remap=%s axis_signs=%s "
        "action_clip=%.3f steps_per_action=%d gripper_threshold=%.3f "
        "gripper_confidence=%.3f auto_calibrate=%s min_cos=%.3f "
        "ik_ratio=[%.2f, %.2f] ik_warn_cooldown=%d max_steps=%d max_tool_seconds=%s",
        cfg.action_scale,
        cfg.axis_remap,
        cfg.axis_signs,
        cfg.action_clip,
        cfg.steps_per_action,
        cfg.gripper_threshold,
        cfg.gripper_confidence,
        cfg.auto_calibrate,
        cfg.auto_calibrate_min_cosine,
        cfg.ik_ratio_low,
        cfg.ik_ratio_high,
        cfg.ik_ratio_warn_cooldown_steps,
        effective_max_steps,
        cfg.max_tool_seconds,
    )
    logger.debug(
        "VLA step log legend: "
        "raw=[dx,dy,dz,gripper] from VLA (unscaled) | "
        "scaled=[dx,dy,dz] after scale+remap (metres, applied to arm) | "
        "sf=adaptive scale factor applied to scaled | "
        "eef=[x,y,z] end-effector world position | "
        "obj=[x,y,z] cube world position | "
        "dist=euclidean eef→obj (metres, should shrink)"
    )

    @tool
    def tool_impl(task: str | None = None) -> str:
        """Execute a VLA-driven manipulation sequence by querying the VLA service."""
        effective_task = task or task_prompt
        if hasattr(arm, "reset_to_initial"):
            arm.reset_to_initial()
        client = VLAClient(server_address, timeout=cfg.request_timeout)
        state = _RunState(
            pre_eef=None,
            pre_obj=None,
            pre_dist=None,
            active_axis_remap=cfg.axis_remap,
            active_axis_signs=cfg.axis_signs,
        )
        ik_ratio_low = cfg.ik_ratio_low
        ik_ratio_high = cfg.ik_ratio_high

        state.pre_eef, state.pre_obj, state.pre_dist = _eef_obj_dist(arm)
        if state.pre_obj and len(state.pre_obj) >= 3:
            state.initial_obj_z = state.pre_obj[2]
        _start_esc = getattr(arm, "env_step_count", None)
        logger.debug(
            "VLA skill start: env_step_count=%s",
            _start_esc if _start_esc is not None else "?",
        )

        def _finish(reason: str, **payload_kwargs: Any) -> str:
            esc = getattr(arm, "env_step_count", None)
            logger.info(
                "VLA skill end: reason=%s steps_applied=%d env_step_count=%s",
                reason,
                state.steps_applied,
                esc if esc is not None else "?",
            )
            return _result_payload(
                reason=reason,
                state=state,
                cfg=cfg,
                **payload_kwargs,
            )

        started = time.monotonic()
        while state.steps_applied < effective_max_steps:
            if (
                cfg.max_tool_seconds is not None
                and time.monotonic() - started >= cfg.max_tool_seconds
            ):
                return _finish(
                    reason="max_tool_seconds",
                    detail=f"tool wall-clock reached {cfg.max_tool_seconds:.2f}s",
                )
            image_bytes = camera.capture_image()
            if cfg.fail_on_placeholder and _camera_vision_unavailable(camera, image_bytes):
                health = getattr(camera, "vision_health", None)
                reason = getattr(health, "degraded_reason", None) or "vision_unavailable"
                return _finish(
                    reason="vision_unavailable",
                    detail=(
                        f"camera vision unavailable ({reason}); "
                        "abort before sending invalid image to VLA"
                    ),
                )
            wrist_image_bytes = None
            capture_wrist = getattr(camera, "capture_wrist_image", None)
            if callable(capture_wrist):
                wrist_image_bytes = capture_wrist()
            try:
                arm_state = getattr(arm, "agent_state", None)
                state_list = arm_state.tolist() if arm_state is not None else None
                chunk = client.predict(
                    image_bytes,
                    effective_task,
                    state=state_list,
                    wrist_image_bytes=wrist_image_bytes,
                )
            except VLAServiceError as exc:
                return str(exc)

            chunk_state = _ChunkState()
            best_min_before_chunk = state.best_min_dist
            chunk_limit = min(len(chunk), cfg.chunk_steps or len(chunk))

            for raw_action in chunk[:chunk_limit]:
                if state.steps_applied >= effective_max_steps:
                    break
                if (
                    cfg.max_tool_seconds is not None
                    and time.monotonic() - started >= cfg.max_tool_seconds
                ):
                    return _finish(
                        reason="max_tool_seconds",
                        detail=f"tool wall-clock reached {cfg.max_tool_seconds:.2f}s",
                    )
                raw_action = list(raw_action)
                state.last_raw_action = raw_action
                raw_mag = math.sqrt(sum(v * v for v in raw_action[:3]))
                if raw_mag > chunk_state.max_raw_mag:
                    chunk_state.max_raw_mag = raw_mag

                dx, dy, dz, sf, _raw_xyz, rot_grip = _apply_raw_action_to_arm(
                    arm, cfg, state, chunk_state, raw_action
                )

                eef, obj, dist = _eef_obj_dist(arm)
                if dist is not None and (state.min_dist is None or dist < state.min_dist):
                    state.min_dist = dist
                state_parity = getattr(arm, "maybe_log_vla_state_parity", None)
                if callable(state_parity):
                    state_parity(state.steps_applied)
                _log_action_telemetry(
                    state.steps_applied,
                    raw_action,
                    (dx, dy, dz),
                    rot_grip,
                    arm,
                    state,
                    eef,
                    dist,
                )
                dist_str = f" dist={dist:.3f}m" if dist is not None else ""
                actual_delta_str, chunk_ik_saturation = _actual_delta_and_ik_ratio(
                    state,
                    eef,
                    dx,
                    dy,
                    dz,
                    ik_ratio_low,
                    ik_ratio_high,
                    cfg.ik_ratio_warn_cooldown_steps,
                    arm,
                )
                chunk_state.ik_saturation = chunk_state.ik_saturation or chunk_ik_saturation

                logger.debug(
                    "VLA step %3d | "
                    "raw=%s "
                    "scaled=[%+.4f,%+.4f,%+.4f] "
                    "sf=%.2f "
                    "actual_delta=%s "
                    "eef=%s obj=%s%s",
                    state.steps_applied,
                    [f"{v:+.3f}" for v in raw_action],
                    dx, dy, dz, sf, actual_delta_str,
                    [f"{value:.3f}" for value in eef] if eef else "?",
                    [f"{value:.3f}" for value in obj] if obj else "?", dist_str,
                )

                if eef:
                    state.prev_eef = eef

                state.steps_applied += 1

                if getattr(arm, "episode_done", False):
                    info = getattr(arm, "last_info", {}) or {}
                    raw_success = info.get("is_success")
                    if raw_success is None:
                        raw_success = info.get("success")
                    is_success: bool | None = None
                    if raw_success is not None:
                        try:
                            import numpy as np

                            if isinstance(raw_success, np.ndarray):
                                is_success = bool(raw_success.flat[0])
                            else:
                                is_success = bool(raw_success)
                        except Exception:
                            is_success = None
                    if is_success is True:
                        logger.info(
                            "Early stop: episode ended with is_success=True "
                            "(task complete); skill finished at step %d",
                            state.steps_applied,
                        )
                        return _finish(
                            reason="task_success",
                            detail="LIBERO episode ended with is_success=True",
                            success=True,
                        )
                    logger.info(
                        "Early stop: episode ended (terminated/truncated); "
                        "skill aborted at step %d",
                        state.steps_applied,
                    )
                    return _finish(
                        reason="episode_done",
                        detail="LIBERO episode terminated or truncated",
                    )

                if dist is not None and (state.min_dist is None or dist < state.min_dist):
                    state.min_dist = dist
                if state.best_min_dist is None or (
                    state.min_dist is not None and state.min_dist < state.best_min_dist
                ):
                    state.best_min_dist = state.min_dist
                if _proximity_grasp_enabled(cfg):
                    proximity_result = _maybe_proximity_grasp(
                        arm, cfg, state, dist, obj
                    )
                    if proximity_result:
                        proximity_detail, proximity_lifted = proximity_result
                        return _finish(
                            reason=(
                                "proximity_grasp" if proximity_lifted else "proximity_no_lift"
                            ),
                            detail=proximity_detail,
                            success=proximity_lifted,
                            grasp_confirmed=proximity_lifted,
                        )
                if _env_task_success(arm, cfg):
                    return _finish(
                        reason="task_success",
                        detail="env task success flag from backend",
                        success=True,
                    )
                if _object_lift_success(arm, cfg, state):
                    return _finish(
                        reason="grasp_lift",
                        detail=(
                            f"object z lift >= {cfg.success_lift_delta:.3f}m "
                            "from initial tracked position"
                        ),
                        success=True,
                        grasp_confirmed=True,
                    )
                fail_reason = _check_failure_heuristics(
                    cfg=cfg,
                    state=state,
                    dist=dist,
                    chunk_ik_saturation=chunk_ik_saturation,
                )
                if fail_reason:
                    return _finish(
                        reason=fail_reason,
                        detail=(
                            f"dist={dist:.3f}m" if dist is not None else "distance unavailable"
                        ),
                    )
                state.pre_eef, state.pre_obj, state.pre_dist = eef, obj, dist

            stagnation_result = _update_stagnation_after_chunk(
                arm,
                cfg,
                state,
                chunk_state,
                best_min_before_chunk,
            )
            if stagnation_result:
                return _finish(
                    reason="stagnation",
                    detail=stagnation_result,
                )

        min_dist_str = (
            f"{state.min_dist:.3f}" if state.min_dist is not None else "unknown"
        )
        return _finish(
            reason="max_inner_steps",
            detail=(
                f"tool reached max inner steps={effective_max_steps} "
                f"(scale={cfg.action_scale}, min_dist={min_dist_str}m)"
            ),
        )

    tool_impl.name = tool_name
    tool_impl.description = tool_description
    return tool_impl

