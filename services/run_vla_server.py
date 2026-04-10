"""Run the VLA HTTP server for the policy selected by VLA_POLICY (see .env)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SUPPORTED = ("octo-small-1.5",)


def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass


def main() -> None:
    _load_dotenv()
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
    if policy == "octo-small-1.5":
        import runpy

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
                "  Windows: pip install -r services/requirements-vla-octo-small-1.5-windows.txt && pip install -e ~/octo --no-deps",
                file=sys.stderr,
            )
            print(
                "  Linux: Octo requirements.txt + pip install -r services/requirements-vla-octo-small-1.5-linux.txt",
                file=sys.stderr,
            )
            print(
                "  If you switched policies and pip conflicts, remove .vla-venv and recreate it.",
                file=sys.stderr,
            )
            print(f"  {exc}", file=sys.stderr)
            sys.exit(3)


if __name__ == "__main__":
    main()
