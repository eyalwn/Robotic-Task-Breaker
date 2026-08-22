import argparse
import sys

from runner import Runner
from utils import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_10")
parser.add_argument("--task", type=int, default=0)
args = parser.parse_args()
configure_logging()
success = Runner(args.suite, args.task).run()
print("PASS" if success else "FAIL")
sys.exit(0 if success else 1)
