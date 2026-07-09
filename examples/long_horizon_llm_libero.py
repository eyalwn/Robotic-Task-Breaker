"""M6 - LLM-emitted plan on LIBERO-Long (LIBERO-10) with full executor + VLA."""
import os
import logging
from dotenv import find_dotenv, load_dotenv
from core.runner import Runner
from sim.libero_backend import LiberoBackend

load_dotenv(find_dotenv())
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
    force=True,
)
# noise filtering from 3rd party
for _noisy in ("openai", "httpcore", "httpx", "trimesh", "matplotlib", "numexpr"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

GOAL = "put the alphabet soup can and the cream cheese box in the basket"
backend = LiberoBackend(task_suite="libero_10", task_idx=0)
result = Runner(trace_dir="traces").run(backend, GOAL)
raise SystemExit(0 if result.success else 1)
