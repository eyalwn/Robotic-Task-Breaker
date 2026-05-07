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


def _parse_optional_float(env_var: str) -> float | None:
    raw = os.environ.get(env_var)
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        logger.warning(
            "%s=%r is invalid; expected a float or empty. Ignoring.",
            env_var,
            raw,
        )
        return None
    if value <= 0.0:
        return None
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
    axis_remap: Optional[List[int]] = None
    axis_signs: Optional[List[float]] = None
    steps_per_action: int = 5
    stagnation_chunks: int = 3
    gripper_threshold: float = 0.5
    # Minimum |raw_gripper| required to allow open/close state changes.
    # This is a raw action-magnitude gate, not a 0-1 model confidence.
    gripper_confidence: float = 1.5
    grasp_dist: float = 0.03
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
    gripper_close_max_dist: float = 0.09
    success_lift_delta: float = 0.02
    fail_on_placeholder: bool = True

    @classmethod
    def from_env(cls) -> "VLAConfig":
        """Construct a VLAConfig from environment variables."""
        action_scale = float(
            os.environ.get("VLA_ACTION_SCALE", str(cls.action_scale))
        )
        action_clip = float(os.environ.get("VLA_ACTION_CLIP", str(cls.action_clip)))
        axis_remap = _parse_axis_remap()
        axis_signs = _parse_axis_signs()
        steps_per_action = int(
            os.environ.get("VLA_STEPS_PER_ACTION", str(cls.steps_per_action))
        )
        stagnation_chunks = int(
            os.environ.get(
                "VLA_STAGNATION_CHUNKS",
                str(cls.stagnation_chunks),
            )
        )
        gripper_threshold = float(
            os.environ.get(
                "VLA_GRIPPER_THRESHOLD",
                str(cls.gripper_threshold),
            )
        )
        gripper_confidence = _parse_gripper_confidence(
            default=cls.gripper_confidence
        )
        grasp_dist = float(
            os.environ.get("VLA_GRASP_DIST", str(cls.grasp_dist))
        )
        request_timeout = float(
            os.environ.get(
                "VLA_REQUEST_TIMEOUT",
                str(cls.request_timeout),
            )
        )
        adaptive_scale = _parse_bool("VLA_ADAPTIVE_SCALE", cls.adaptive_scale)
        adaptive_ref_dist = float(
            os.environ.get(
                "VLA_ADAPTIVE_REF_DIST",
                str(cls.adaptive_ref_dist),
            )
        )
        adaptive_min_factor = float(
            os.environ.get(
                "VLA_ADAPTIVE_MIN_FACTOR",
                str(cls.adaptive_min_factor),
            )
        )
        auto_calibrate = _parse_bool("VLA_AUTO_CALIBRATE", cls.auto_calibrate)
        auto_calibrate_min_cosine = float(
            os.environ.get(
                "VLA_AUTO_CALIBRATE_MIN_COSINE",
                str(cls.auto_calibrate_min_cosine),
            )
        )
        ik_ratio_low = float(
            os.environ.get("VLA_IK_RATIO_LOW", str(cls.ik_ratio_low))
        )
        ik_ratio_high = float(
            os.environ.get("VLA_IK_RATIO_HIGH", str(cls.ik_ratio_high))
        )
        ik_ratio_warn_cooldown_steps = int(
            os.environ.get(
                "VLA_IK_RATIO_WARN_COOLDOWN_STEPS",
                str(cls.ik_ratio_warn_cooldown_steps),
            )
        )
        max_inner_steps = int(
            os.environ.get("VLA_MAX_INNER_STEPS", str(cls.max_inner_steps))
        )
        max_tool_seconds = _parse_max_tool_seconds(
            "VLA_MAX_TOOL_SECONDS",
            cls.max_tool_seconds,
        )
        fail_dist_threshold = float(
            os.environ.get("VLA_FAIL_DIST_THRESHOLD", str(cls.fail_dist_threshold))
        )
        fail_dist_consecutive_steps = int(
            os.environ.get(
                "VLA_FAIL_DIST_CONSECUTIVE_STEPS",
                str(cls.fail_dist_consecutive_steps),
            )
        )
        fail_worsening_steps = int(
            os.environ.get(
                "VLA_FAIL_WORSENING_STEPS",
                str(cls.fail_worsening_steps),
            )
        )
        fail_worsening_margin = float(
            os.environ.get(
                "VLA_FAIL_WORSENING_MARGIN",
                str(cls.fail_worsening_margin),
            )
        )
        fail_ik_saturation_steps = int(
            os.environ.get(
                "VLA_FAIL_IK_SATURATION_STEPS",
                str(cls.fail_ik_saturation_steps),
            )
        )
        sim_pickcube_gripper_profile = _parse_bool(
            "VLA_SIM_PICKCUBE_GRIPPER_PROFILE",
            cls.sim_pickcube_gripper_profile,
        )
        gripper_close_max_dist = float(
            os.environ.get(
                "VLA_GRIPPER_CLOSE_MAX_DIST",
                str(cls.gripper_close_max_dist),
            )
        )
        success_lift_delta = float(
            os.environ.get("VLA_SUCCESS_LIFT_DELTA", str(cls.success_lift_delta))
        )
        fail_on_placeholder = _parse_bool(
            "VLA_FAIL_ON_PLACEHOLDER", cls.fail_on_placeholder
        )

        return cls(
            action_scale=action_scale,
            action_clip=action_clip,
            axis_remap=axis_remap,
            axis_signs=axis_signs,
            steps_per_action=steps_per_action,
            stagnation_chunks=stagnation_chunks,
            gripper_threshold=gripper_threshold,
            gripper_confidence=gripper_confidence,
            grasp_dist=grasp_dist,
            request_timeout=request_timeout,
            adaptive_scale=adaptive_scale,
            adaptive_ref_dist=adaptive_ref_dist,
            adaptive_min_factor=adaptive_min_factor,
            auto_calibrate=auto_calibrate,
            auto_calibrate_min_cosine=auto_calibrate_min_cosine,
            ik_ratio_low=ik_ratio_low,
            ik_ratio_high=ik_ratio_high,
            ik_ratio_warn_cooldown_steps=ik_ratio_warn_cooldown_steps,
            max_inner_steps=max_inner_steps,
            max_tool_seconds=max_tool_seconds,
            fail_dist_threshold=fail_dist_threshold,
            fail_dist_consecutive_steps=fail_dist_consecutive_steps,
            fail_worsening_steps=fail_worsening_steps,
            fail_worsening_margin=fail_worsening_margin,
            fail_ik_saturation_steps=fail_ik_saturation_steps,
            sim_pickcube_gripper_profile=sim_pickcube_gripper_profile,
            gripper_close_max_dist=gripper_close_max_dist,
            success_lift_delta=success_lift_delta,
            fail_on_placeholder=fail_on_placeholder,
        )