from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)


def _parse_gripper_confidence(
    env_var: str = "VLA_GRIPPER_CONFIDENCE",
    default: float = 0.5,
) -> float:
    """Parse minimum |raw_gripper| gate and guard against absurd values."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is invalid; expected a float. Using default=%.3f.",
            env_var,
            raw,
            default,
        )
        return default

    if value > 2.0:
        logger.warning(
            "%s=%.3f is out of expected range (<= 2.0). Using default=%.3f.",
            env_var,
            value,
            default,
        )
        return default
    return value


def _parse_axis_remap(env_var: str = "VLA_AXIS_REMAP") -> Optional[List[int]]:
    """Parse VLA_AXIS_REMAP env var into an index list for numpy-style remapping.

    Format: comma-separated indices for output [x, y, z], e.g. "0,2,1" swaps y/z.
    Returns None if unset (no remapping).
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    try:
        indices = [int(i) for i in raw.split(",")]
        if len(indices) != 3:
            raise ValueError
        return indices
    except ValueError:
        logger.warning(
            "VLA_AXIS_REMAP=%r is invalid; expected 3 comma-separated ints. Ignoring.",
            raw,
        )
        return None


def _parse_axis_signs(env_var: str = "VLA_AXIS_SIGNS") -> Optional[List[float]]:
    """Parse VLA_AXIS_SIGNS env var into sign multipliers.

    Format: comma-separated +1/-1 values for [x, y, z], e.g. "1,-1,1" flips y.
    Returns None if unset (no sign flip).
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    try:
        signs = [float(s) for s in raw.split(",")]
        if len(signs) != 3:
            raise ValueError
        return signs
    except ValueError:
        logger.warning(
            "VLA_AXIS_SIGNS=%r is invalid; expected 3 comma-separated floats. Ignoring.",
            raw,
        )
        return None


def _parse_gripper_mode(env_var: str = "VLA_GRIPPER_MODE", default: str = "delta_sign") -> str:
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"delta_sign", "absolute_open_close"}:
        return value
    logger.warning(
        "%s=%r is invalid; expected 'delta_sign' or 'absolute_open_close'. Using default=%r.",
        env_var,
        raw,
        default,
    )
    return default


def _parse_bool(env_var: str, default: bool) -> bool:
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    logger.warning(
        "%s=%r is invalid; expected a boolean. Using default=%s.",
        env_var,
        raw,
        default,
    )
    return default


def _ef(var: str, default) -> str:
    return os.environ.get(var, str(default))


def _parse_chunk_steps(env_var: str = "VLA_CHUNK_STEPS", default: int = 0) -> int:
    """Actions to execute per VLA predict before re-querying (0 = full chunk)."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        logger.warning(
            "%s=%r is invalid; expected an int. Using default=%d.",
            env_var,
            raw,
            default,
        )
        return default
    if value < 0:
        logger.warning(
            "%s=%d is negative; using default=%d.",
            env_var,
            value,
            default,
        )
        return default
    return value


def _parse_max_tool_seconds(
    env_var: str,
    default_when_unset: float | None,
) -> float | None:
    """Wall-clock cap for one VLA tool call.

    - Env unset → ``default_when_unset`` (class default, typically ~90s for slow CPUs).
    - ``<= 0`` or empty string → ``None`` (no cap).
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default_when_unset
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        logger.warning(
            "%s=%r is invalid; expected a float, empty, or <=0 to disable. "
            "Using default=%s.",
            env_var,
            raw,
            default_when_unset,
        )
        return default_when_unset
    if value <= 0.0:
        return None
    return value


@dataclass
class VLAConfig:
    action_scale: float = 0.05
    action_clip: float = 2.0
    axis_remap: list[int] | None = None
    axis_signs: list[float] | None = None
    steps_per_action: int = 5
    # 0 = execute full predicted chunk before next predict (PickCube default).
    chunk_steps: int = 0
    stagnation_chunks: int = 3
    gripper_threshold: float = 0.5
    # Minimum |raw_gripper| required to allow open/close state changes.
    # This is a raw action-magnitude gate, not a 0-1 model confidence.
    gripper_confidence: float = 1.5
    grasp_dist: float = 0.03
    # When false, or on OpenPI LIBERO raw-action path, skip proximity-based early exit.
    proximity_grasp: bool = True
    request_timeout: float = 60.0
    adaptive_scale: bool = True
    adaptive_ref_dist: float = 0.15
    adaptive_min_factor: float = 0.2
    auto_calibrate: bool = False
    auto_calibrate_min_cosine: float = 0.7
    ik_ratio_low: float = 0.3
    ik_ratio_high: float = 3.0
    ik_ratio_warn_cooldown_steps: int = 20
    max_inner_steps: int = 60
    # Modest default so CPU-bound VLA + sim runs get more than a few steps before
    # max_tool_seconds; set VLA_MAX_TOOL_SECONDS=0 to disable the cap.
    max_tool_seconds: float | None = 25.0
    fail_dist_threshold: float = 0.22
    fail_dist_consecutive_steps: int = 20
    fail_worsening_steps: int = 20
    fail_worsening_margin: float = 0.03
    fail_ik_saturation_steps: int = 25
    sim_pickcube_gripper_profile: bool = True
    # delta_sign: positive raw → open, negative → close (ManiSkill/Octo).
    # absolute_open_close: raw in [0, 1] with 0=open, 1=close (LIBERO pi05_libero).
    gripper_mode: str = "delta_sign"
    gripper_close_max_dist: float = 0.09
    success_lift_delta: float = 0.02
    # When true, per-skill VLA may exit on env.evaluate() / last_info success flags
    # (ManiSkill full-task). LIBERO multi-step plans should set false.
    use_env_task_success: bool = True
    fail_on_placeholder: bool = True

    @classmethod
    def from_env(cls) -> "VLAConfig":
        """Construct a VLAConfig from environment variables."""
        f = _ef
        b = _parse_bool
        return cls(
            action_scale=float(f("VLA_ACTION_SCALE", cls.action_scale)),
            action_clip=float(f("VLA_ACTION_CLIP", cls.action_clip)),
            axis_remap=_parse_axis_remap(),
            axis_signs=_parse_axis_signs(),
            steps_per_action=int(f("VLA_STEPS_PER_ACTION", cls.steps_per_action)),
            chunk_steps=_parse_chunk_steps(),
            stagnation_chunks=int(f("VLA_STAGNATION_CHUNKS", cls.stagnation_chunks)),
            gripper_threshold=float(f("VLA_GRIPPER_THRESHOLD", cls.gripper_threshold)),
            gripper_confidence=_parse_gripper_confidence(default=cls.gripper_confidence),
            grasp_dist=float(f("VLA_GRASP_DIST", cls.grasp_dist)),
            proximity_grasp=b("VLA_PROXIMITY_GRASP", cls.proximity_grasp),
            request_timeout=float(f("VLA_REQUEST_TIMEOUT", cls.request_timeout)),
            adaptive_scale=b("VLA_ADAPTIVE_SCALE", cls.adaptive_scale),
            adaptive_ref_dist=float(f("VLA_ADAPTIVE_REF_DIST", cls.adaptive_ref_dist)),
            adaptive_min_factor=float(f("VLA_ADAPTIVE_MIN_FACTOR", cls.adaptive_min_factor)),
            auto_calibrate=b("VLA_AUTO_CALIBRATE", cls.auto_calibrate),
            auto_calibrate_min_cosine=float(
                f("VLA_AUTO_CALIBRATE_MIN_COSINE", cls.auto_calibrate_min_cosine)
            ),
            ik_ratio_low=float(f("VLA_IK_RATIO_LOW", cls.ik_ratio_low)),
            ik_ratio_high=float(f("VLA_IK_RATIO_HIGH", cls.ik_ratio_high)),
            ik_ratio_warn_cooldown_steps=int(
                f("VLA_IK_RATIO_WARN_COOLDOWN_STEPS", cls.ik_ratio_warn_cooldown_steps)
            ),
            max_inner_steps=int(f("VLA_MAX_INNER_STEPS", cls.max_inner_steps)),
            max_tool_seconds=_parse_max_tool_seconds("VLA_MAX_TOOL_SECONDS", cls.max_tool_seconds),
            fail_dist_threshold=float(f("VLA_FAIL_DIST_THRESHOLD", cls.fail_dist_threshold)),
            fail_dist_consecutive_steps=int(
                f("VLA_FAIL_DIST_CONSECUTIVE_STEPS", cls.fail_dist_consecutive_steps)
            ),
            fail_worsening_steps=int(f("VLA_FAIL_WORSENING_STEPS", cls.fail_worsening_steps)),
            fail_worsening_margin=float(f("VLA_FAIL_WORSENING_MARGIN", cls.fail_worsening_margin)),
            fail_ik_saturation_steps=int(
                f("VLA_FAIL_IK_SATURATION_STEPS", cls.fail_ik_saturation_steps)
            ),
            sim_pickcube_gripper_profile=b("VLA_SIM_PICKCUBE_GRIPPER_PROFILE", cls.sim_pickcube_gripper_profile),
            gripper_mode=_parse_gripper_mode(),
            gripper_close_max_dist=float(
                f("VLA_GRIPPER_CLOSE_MAX_DIST", cls.gripper_close_max_dist)
            ),
            success_lift_delta=float(f("VLA_SUCCESS_LIFT_DELTA", cls.success_lift_delta)),
            use_env_task_success=b(
                "VLA_USE_ENV_TASK_SUCCESS", cls.use_env_task_success
            ),
            fail_on_placeholder=b("VLA_FAIL_ON_PLACEHOLDER", cls.fail_on_placeholder),
        )