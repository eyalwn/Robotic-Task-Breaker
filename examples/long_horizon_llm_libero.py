"""M6 - LLM-emitted plan on LIBERO-Long (LIBERO-10) with full executor + VLA."""
import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_examples = Path(__file__).resolve().parent
_src = _examples.parent / "src"
for path in (_src, _examples):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

load_dotenv(find_dotenv())
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
    force=True,
)
for _h in logging.root.handlers:
    if isinstance(_h, logging.StreamHandler) and not isinstance(
        _h, logging.FileHandler
    ):
        _h.setLevel(logging.INFO)
for _noisy in ("openai", "httpcore", "httpx", "trimesh", "matplotlib", "numexpr"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from core.runner import Runner
from sim.libero_backend import LiberoBackend

GOAL = "put the alphabet soup can and the cream cheese box in the basket"


def main() -> None:
    backend = LiberoBackend(
        task_suite=os.environ.get("LIBERO_TASK_SUITE", "libero_10"),
    )
    result = Runner(
        trace_dir=os.environ.get("LONGOR_TRACE_DIR", "traces"),
    ).run(backend, GOAL)
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
