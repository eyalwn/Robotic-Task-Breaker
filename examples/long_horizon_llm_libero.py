from runner import Runner
from utils import configure_logging

configure_logging()
GOAL = "put the alphabet soup can and the tomato sauce in the basket"
Runner(GOAL, task_suite="libero_10", task_idx=0).run()
print("DONE")