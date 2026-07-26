"""LIBERO integration tests (VLA server + sim required).

Run: pytest -m integration tests/test_libero_integration.py
"""
from __future__ import annotations

import os

import pytest
import requests

from core.plan import (
    AlwaysContinueEvaluator,
    StaticPlanner,
    grab_object_plan,
    resolve_libero_grasp_object,
)
from core.runner import Runner
from sim.libero_backend import LiberoBackend
from sim.libero_smoke import run_grasp_smoke, run_place_smoke, run_sequential_smoke
from utils import load_env

pytestmark = pytest.mark.integration

_DEFAULT_VLA_URL = "http://localhost:8765"


def _libero_deps_available() -> bool:
    try:
        import lerobot.envs.factory  # noqa: F401
        import libero.libero  # noqa: F401
    except ImportError:
        return False
    return True


def _vla_url() -> str:
    port = os.environ.get("VLA_PORT")
    if port:
        return f"http://localhost:{port}"
    return _DEFAULT_VLA_URL


@pytest.fixture(scope="module", autouse=True)
def _load_env_fixture() -> None:
    load_env()


@pytest.fixture(scope="module")
def require_libero_and_vla() -> str:
    if not _libero_deps_available():
        pytest.skip("LIBERO deps not installed (see deps/requirements-libero.txt)")
    url = _vla_url()
    try:
        requests.get(f"{url}/health", timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip(f"VLA service not reachable at {url}")
    return url


def test_grasp_smoke(require_libero_and_vla: str) -> None:
    del require_libero_and_vla
    assert run_grasp_smoke() == 0


def test_place_smoke(require_libero_and_vla: str) -> None:
    del require_libero_and_vla
    assert run_place_smoke() == 0


def test_sequential_vla_smoke(require_libero_and_vla: str) -> None:
    del require_libero_and_vla
    assert run_sequential_smoke() == 0


def test_handcrafted_libero_pick(require_libero_and_vla: str) -> None:
    del require_libero_and_vla
    backend = LiberoBackend(task_suite="libero_object", task_idx=0)
    object_name = resolve_libero_grasp_object(backend.list_objects())
    plan = grab_object_plan(object_name)
    result = Runner(max_replans=0).run(
        backend,
        plan.goal,
        planner=StaticPlanner(plan),
        feedback=AlwaysContinueEvaluator(),
    )
    assert result.status == "done"
