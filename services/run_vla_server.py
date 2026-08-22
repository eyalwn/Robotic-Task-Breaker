"""Launch the VLA HTTP server for smolvla / pi05."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import config

_SERVICE_FILES = {
    "pi05": "pi05_service.py",
    "smolvla": "smolvla_service.py",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Longor VLA policy HTTP service."
    )
    parser.add_argument(
        "--policy",
        choices=config.VLA_POLICIES,
        default=None,
        help=f"policy backend (default: config.VLA_POLICY={config.VLA_POLICY!r})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    policy = args.policy or config.VLA_POLICY
    if policy not in config.VLA_POLICIES:
        supported = ", ".join(config.VLA_POLICIES)
        print(
            f"error: unknown policy {policy!r}. Supported: {supported}.",
            file=sys.stderr,
        )
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
        if policy == "pi05":
            print(
                "error: OpenPI (Pi0.5) or HTTP deps not importable in this venv.",
                file=sys.stderr,
            )
            print(
                "  Linux + NVIDIA: install OpenPI from "
                "https://github.com/Physical-Intelligence/openpi "
                "(uv sync / pip install -e . per upstream), then:",
                file=sys.stderr,
            )
            print(
                "    pip install -r deps/requirements-vla-pi05-linux.txt && pip install -e .",
                file=sys.stderr,
            )
        else:
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
                "  Default checkpoint: lerobot/smolvla_vlabench "
                "(override with SMOLVLA_CHECKPOINT).",
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
