"""Manual live integration test for executor black-box skill execution."""
from __future__ import annotations

import pytest

from executor import Executor, StopReason
from plan import Plan
from vla_client import VLAClient, VLAError

SUITE = "libero_object"
TASK_IDX = 0
INIT_STATE_IDX = 0
OBJECT_NAME = "alphabet_soup_1"
TARGET_NAME = "basket_1_contain_region"


def _manual_only(pytestconfig: pytest.Config) -> None:
    """Skip unless the caller explicitly selects ``-m integration``."""
    if (pytestconfig.getoption("-m") or "").strip() != "integration":
        pytest.skip(
            "manual live test; run with: "
            "pytest -m integration tests/test_executor_live_libero.py -q -s"
        )


def _grab_place_plan() -> Plan:
    """Build a grab+place plan for live black-box execution."""
    return [
        {"skill": "grab", "object_name": OBJECT_NAME},
        {
            "skill": "place",
            "object_name": OBJECT_NAME,
            "target_name": TARGET_NAME,
            "position": "inside",
        },
    ]


@pytest.mark.integration
def test_executor_grab_place_reaches_libero_object_task0(
    pytestconfig: pytest.Config,
) -> None:
    """Run one live executor plan and gate pass/fail on ``env.success()``."""
    _manual_only(pytestconfig)

    try:
        from envs.libero_env import LiberoEnv
    except ImportError as exc:
        pytest.skip(f"LIBERO unavailable: {exc}")

    vla = VLAClient()
    try:
        health = vla.health()
    except VLAError as exc:
        pytest.skip(f"VLA server unavailable: {exc}")
    print(f"vla service       {health}")

    env = LiberoEnv("libero_object", 0, init_state_idx=INIT_STATE_IDX)
    try:
        goal = env.task_description
        goal_lower = goal.lower()
        if "alphabet soup" not in goal_lower or "basket" not in goal_lower:
            pytest.skip(
                "task text drifted; this test assumes a basket-placement task "
                f"for {SUITE} task {TASK_IDX}, got {goal!r}"
            )

        plan = _grab_place_plan()
        print(f"suite             {SUITE!r}")
        print(f"task              {TASK_IDX}")
        print(f"init_state_idx    {INIT_STATE_IDX}")
        print(f"task_description  {goal!r}")
        print(
            "plan_steps        "
            f"grab({OBJECT_NAME!r}) then place({OBJECT_NAME!r}, "
            f"{TARGET_NAME!r}, inside)"
        )

        executor = Executor(env=env)
        executor._vla = vla
        result = executor.run(plan)
        success = env.success()
        steps = env.steps

        print(f"execution_reason  {result.reason}")
        print(f"env_steps         {steps}")
        print(f"env_success       {success}")
    finally:
        env.close()

    assert result.reason == StopReason.PLAN_DONE
    assert success, (
        "env.success() is false after one grab+place black-box run; "
        f"steps={steps}, reason={result.reason}"
    )
