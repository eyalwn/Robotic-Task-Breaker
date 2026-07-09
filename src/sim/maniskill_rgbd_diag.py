# Exit codes: 0=ok; 2=mani_skill import failed; 3=Vulkan DeviceLost; 4=gym.make/reset failed; 5=rgbd obs without RGB tensors.
"""Single-shot ManiSkill RGBD probe: prints diag_config/diag_result/diag_status on stdout."""
from __future__ import annotations

import os

import gymnasium as gym
from dotenv import find_dotenv, load_dotenv

from sim.maniskill_backend import _default_sim_backend


def _has_rgb(obs) -> bool:
    if not isinstance(obs, dict):
        return False

    def _walk(value) -> bool:
        try:
            import numpy as np

            arr = np.asarray(value)
            if arr.ndim >= 3 and arr.shape[-1] == 3:
                return True
        except Exception:
            pass
        if isinstance(value, dict):
            return any(_walk(v) for v in value.values())
        return False

    return _walk(obs)


def run() -> int:
    load_dotenv(find_dotenv())
    try:
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(f"diag_status=error reason=missing_maniskill detail={exc}")
        return 2

    env_id = os.environ.get("MANISKILL_ENV", "").strip()
    if not env_id:
        print("diag_status=error reason=missing_MANISKILL_ENV detail=set MANISKILL_ENV e.g. PickCube-v1")
        return 4
    obs_mode = os.environ.get("MANISKILL_OBS_MODE", "rgbd")
    render_mode = "human" if os.environ.get("DISPLAY_ENV", "0") == "1" else "rgb_array"
    render_backend = os.environ.get("MANISKILL_RENDER_BACKEND", "").strip() or None
    sim_backend = os.environ.get("MANISKILL_SIM_BACKEND", "").strip() or _default_sim_backend()
    control_mode = os.environ.get("MANISKILL_CONTROL_MODE", "pd_joint_delta_pos")
    kwargs = {
        "obs_mode": obs_mode,
        "render_mode": render_mode,
        "control_mode": control_mode,
    }
    if render_backend is not None:
        kwargs["render_backend"] = render_backend
    if sim_backend is not None:
        kwargs["sim_backend"] = sim_backend
    print(
        f"diag_config env={env_id} obs_mode={obs_mode} render_mode={render_mode} "
        f"render_backend={render_backend or 'default'} sim_backend={sim_backend or 'default'}"
    )

    try:
        env = gym.make(env_id, **kwargs)
        obs, _ = env.reset(seed=0)
    except RuntimeError as exc:
        detail = str(exc)
        if "DeviceLost" in detail or "device lost" in detail.lower():
            print(f"diag_status=error reason=vulkan_device_lost detail={detail}")
            return 3
        print(f"diag_status=error reason=gym_make_failed detail={detail}")
        return 4
    except Exception as exc:
        print(f"diag_status=error reason=gym_make_failed detail={exc}")
        return 4

    has_rgb = _has_rgb(obs)
    print(f"diag_result has_rgb={int(has_rgb)} obs_type={type(obs).__name__}")
    if obs_mode == "rgbd" and not has_rgb:
        print("diag_status=error reason=rgbd_missing_rgb")
        return 5

    print("diag_status=ok")
    return 0
