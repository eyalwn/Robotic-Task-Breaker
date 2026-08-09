from runner import Runner
from utils import configure_logging

configure_logging()
Runner(task_suite="libero_10", task_idx=0).run()
print("DONE")
