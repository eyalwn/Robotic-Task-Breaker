"""Run the VLA HTTP server for the policy selected by VLA_POLICY (see .env)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SUPPORTED = ("octo-small-1.5", "pi05", "smolvla")


def main() -> None:
    try:
        from utils import load_env

        load_env(usecwd=True)
    except ImportError:
        pass
    policy = (os.environ.get("VLA_POLICY") or "").strip()
    if not policy:
        print(
            "error: VLA_POLICY is unset. Set it in .env (e.g. VLA_POLICY=octo-small-1.5).",
            file=sys.stderr,
        )
        sys.exit(2)
    if policy not in _SUPPORTED:
        print(
            f"error: unknown VLA_POLICY={policy!r}. Supported: {', '.join(_SUPPORTED)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    here = Path(__file__).resolve().parent
    import runpy

    if policy == "octo-small-1.5":
        target = here / "octo_service.py"
        if not target.is_file():
            print(f"error: missing {target}", file=sys.stderr)
            sys.exit(1)
        try:
            runpy.run_path(str(target), run_name="__main__")
        except ImportError as exc:
            print(
                "error: Octo/JAX (or HTTP) deps not importable in this venv.",
                file=sys.stderr,
            )
            print(
                "  Windows: pip install -r deps/requirements-vla-octo-small-1.5-windows.txt "
                "&& pip install -e ~/octo --no-deps && pip install -e .",
                file=sys.stderr,
            )
            print(
                "  Linux: Octo requirements.txt + pip install -r "
                "deps/requirements-vla-octo-small-1.5-linux.txt && pip install -e .",
                file=sys.stderr,
            )
            print(
                "  If you switched policies and pip conflicts, remove .vla-venv and recreate it.",
                file=sys.stderr,
            )
            print(f"  {exc}", file=sys.stderr)
            sys.exit(3)
        return

    if policy == "pi05":
        target = here / "pi05_service.py"
        if not target.is_file():
            print(f"error: missing {target}", file=sys.stderr)
            sys.exit(1)
        try:
            runpy.run_path(str(target), run_name="__main__")
        except ImportError as exc:
            print(
                "error: OpenPI (Pi0.5) or HTTP deps not importable in this venv.",
                file=sys.stderr,
            )
            print(
                "  Linux + NVIDIA: install OpenPI from https://github.com/Physical-Intelligence/openpi "
                "(uv sync / pip install -e . per upstream), then:",
                file=sys.stderr,
            )
            print(
                "    pip install -r deps/requirements-vla-pi05-linux.txt && pip install -e .",
                file=sys.stderr,
            )
            print(
                "  Pi0.5 is not supported on Windows. If you switched policies, recreate .vla-venv.",
                file=sys.stderr,
            )
            print(f"  {exc}", file=sys.stderr)
            sys.exit(3)
        return

    if policy == "smolvla":
        target = here / "smolvla_service.py"
        if not target.is_file():
            print(f"error: missing {target}", file=sys.stderr)
            sys.exit(1)
        try:
            runpy.run_path(str(target), run_name="__main__")
        except ImportError as exc:
            print(
                "error: SmolVLA (LeRobot) or HTTP deps not importable in this venv.",
                file=sys.stderr,
            )
            print(
                "  Linux + NVIDIA (T4-class OK):",
                file=sys.stderr,
            )
            print(
                "    pip install -r deps/requirements-vla-smolvla-linux.txt && pip install -e .",
                file=sys.stderr,
            )
            print(
                "  Default checkpoint: HuggingFaceVLA/smolvla_libero "
                "(override with SMOLVLA_CHECKPOINT).",
                file=sys.stderr,
            )
            print(
                "  If you switched policies and pip conflicts, remove .vla-venv and recreate it.",
                file=sys.stderr,
            )
            print(f"  {exc}", file=sys.stderr)
            sys.exit(3)
        return


if __name__ == "__main__":
    main()
