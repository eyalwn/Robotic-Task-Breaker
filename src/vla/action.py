from __future__ import annotations

from typing import List, Sequence, Tuple

from .config import VLAConfig


def process_raw_action(
    raw_xyz: List[float],
    cfg: VLAConfig,
    axis_remap: Sequence[int] | None = None,
    axis_signs: Sequence[float] | None = None,
) -> Tuple[float, float, float]:
    """Clip, remap, sign-flip, and scale a raw [dx, dy, dz] vector."""
    if len(raw_xyz) < 3:
        raise ValueError("raw_xyz must have at least 3 elements")

    def _clip(v: float) -> float:
        return max(-cfg.action_clip, min(cfg.action_clip, v))

    x, y, z = [_clip(v) for v in raw_xyz[:3]]
    xyz = [x, y, z]

    remap = cfg.axis_remap if axis_remap is None else axis_remap
    signs = cfg.axis_signs if axis_signs is None else axis_signs

    if remap is not None:
        xyz = [xyz[i] for i in remap]
    if signs is not None:
        xyz = [v * s for v, s in zip(xyz, signs)]

    scaled = [v * cfg.action_scale for v in xyz]
    return scaled[0], scaled[1], scaled[2]


def _normalize_gripper_absolute(clipped: float) -> float:
    """Map raw gripper to [0, 1] for absolute_open_close (0=open, 1=close)."""
    if -1.0 <= clipped <= 1.0:
        return (clipped + 1.0) / 2.0
    return max(0.0, min(1.0, clipped))


def requested_gripper_state(raw_gripper: float, cfg: VLAConfig) -> str:
    """Map a raw gripper channel to the requested open/close state."""
    clipped = max(-cfg.action_clip, min(cfg.action_clip, raw_gripper))
    if cfg.gripper_mode == "absolute_open_close":
        normalized = _normalize_gripper_absolute(clipped)
        return "close" if normalized >= cfg.gripper_threshold else "open"
    return "open" if clipped > 0 else "close"


def should_flip_gripper(raw_gripper: float, current_state: str, cfg: VLAConfig) -> bool:
    """Return True when raw gripper signal is strong enough to flip state.

  In ``delta_sign`` mode a flip requires:
    - requested state differs from ``current_state``
    - ``abs(raw_gripper)`` exceeds ``gripper_threshold``
    - ``abs(raw_gripper)`` also exceeds ``gripper_confidence``

  In ``absolute_open_close`` mode (LIBERO pi05: 0=open, 1=close) the requested
  state follows ``gripper_threshold`` and flips whenever it differs from current.
    """
    clipped_gripper = max(-cfg.action_clip, min(cfg.action_clip, raw_gripper))
    requested_state = requested_gripper_state(clipped_gripper, cfg)
    if requested_state == current_state:
        return False

    if cfg.gripper_mode == "absolute_open_close":
        return True

    magnitude = abs(clipped_gripper)
    if magnitude < cfg.gripper_threshold:
        return False
    if magnitude <= cfg.gripper_confidence:
        return False

    return True

