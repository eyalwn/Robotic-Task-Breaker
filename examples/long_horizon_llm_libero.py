"""M6 - LLM-emitted plan on LIBERO-Long (LIBERO-10) with full executor + VLA."""
from core.runner import Runner
from sim.libero_backend import LiberoBackend
from utils import configure_logging, load_env

load_env()
configure_logging()

GOAL = "put the alphabet soup can and the cream cheese box in the basket"
backend = LiberoBackend(task_suite="libero_10", task_idx=0)
result = Runner(trace_dir="traces").run(backend, GOAL)
raise SystemExit(0 if result.success else 1)
