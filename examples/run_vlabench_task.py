import argparse

from runner import Runner
from utils import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="select_fruit")
args = parser.parse_args()
configure_logging()
Runner(args.task).run()
print("DONE")
