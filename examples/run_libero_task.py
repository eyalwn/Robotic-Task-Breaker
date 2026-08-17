import argparse

from runner import Runner
from utils import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_10")
parser.add_argument("--task", type=int, default=0)
args = parser.parse_args()
configure_logging()
Runner(args.suite, args.task).run()
print("DONE")
