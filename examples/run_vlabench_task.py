import argparse
import sys

from runner import Runner
from utils import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="select_fruit")
args = parser.parse_args()
configure_logging()
success = Runner(args.task).run()
verdict = "PASS" if success else "FAIL"
print(f"{verdict} task={args.task!r} success={success}")
sys.exit(0 if success else 1)
