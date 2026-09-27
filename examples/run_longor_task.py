import argparse
import sys

from runner import Runner
from utils import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open_drawer")
parser.add_argument("--task-idx", type=int, default=0)
args = parser.parse_args()
configure_logging()
try:
    success = Runner(args.task, args.task_idx).run()
except Exception as exc:
    print(f"FAIL task={args.task!r} task_idx={args.task_idx} error={exc!r}")
    sys.exit(1)
verdict = "PASS" if success else "FAIL"
print(f"{verdict} task={args.task!r} task_idx={args.task_idx} success={success}")
sys.exit(0 if success else 1)
