"""Launch the VLA HTTP server for GR00T."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import config

_SERVICE_FILES = {"groot": "groot_service.py"}
_SUPPORTED_POLICIES = tuple(_SERVICE_FILES)


def main() -> None:
    policy = config.VLA_POLICY
    if policy not in _SUPPORTED_POLICIES:
        supported = ", ".join(_SUPPORTED_POLICIES)
        print(f"error: unknown policy {policy!r}. Supported: {supported}.",
              file=sys.stderr)
        sys.exit(1)

    service_name = _SERVICE_FILES[policy]
    here = Path(__file__).resolve().parent
    target = here / service_name
    if not target.is_file():
        print(f"error: missing {target}", file=sys.stderr)
        sys.exit(1)

    try:
        runpy.run_path(str(target), run_name="__main__")
    except ImportError as exc:
        print("error: Isaac-GR00T upstream client or HTTP deps not "
              "importable.", file=sys.stderr)
        print("  Linux + NVIDIA: install Isaac-GR00T n1d6 and start upstream "
              "run_gr00t_server.py on port 5555.", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
