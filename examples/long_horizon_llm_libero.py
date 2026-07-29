from runner import Runner
from utils import configure_logging

configure_logging()
GOAL = "put the alphabet soup can and the cream cheese box in the basket"
result = Runner(GOAL, task_suite="libero_10", task_idx=0).run()
raise SystemExit(0 if result.success else 1)