# THROWAWAY Colab bisect: video black/static — delete after use
from __future__ import annotations

import argparse
import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PROMPT = "open the left drawer"
TASK, TASK_IDX = "open_drawer", 0
N_CHUNKS, CHUNK_STEPS = 3, 8
TOTAL_STEPS = N_CHUNKS * CHUNK_STEPS
_WATCH = ("cv2", "torch", "langchain", "openai", "PIL", "requests")
_E_EXTRA = ("openai", "langchain_openai", "httpx", "tiktoken")


def report(env: object, step: int, post: bool = False) -> None:
    obs = env._last_obs
    step_label = f"{step}+post" if post else str(step)
    for k in sorted(obs.keys()):
        if not k.startswith("video."):
            continue
        arr = np.asarray(obs[k]).astype(float)
        mean, mx = float(np.mean(arr)), int(np.max(arr))
        line = f"step={step_label} key={k} mean={mean:.2f} max={mx}"
        if mean < 2.0:
            line += " BLACK"
        print(line, flush=True)


def gpu(tag: str) -> None:
    cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"GPU[{tag}] {out.stdout.strip()}", flush=True)
    except Exception as exc:
        print(f"GPU[{tag}] error: {exc}", flush=True)


def _banner() -> None:
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}", flush=True)
    print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')!r}", flush=True)
    print(f"loaded_modules={sorted(m for m in _WATCH if m in sys.modules)}", flush=True)


def _loaded_watch(extra: tuple[str, ...] = ()) -> None:
    names = _WATCH + extra
    print(f"loaded_modules={sorted(m for m in names if m in sys.modules)}", flush=True)


def _load_chunks(path: str) -> list[list[list[float]]]:
    found: list[list[list[float]]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if "payload = {'actions':" not in line:
                continue
            payload = ast.literal_eval(line.split("payload = ", 1)[1].strip())
            found.append(payload["actions"])
    if len(found) < N_CHUNKS:
        raise RuntimeError(f"log has {len(found)} action chunks, need {N_CHUNKS}")
    return found[:N_CHUNKS]


def _tag_steps(step: int) -> None:
    if step == 1:
        gpu("after_step_1")
    elif step == TOTAL_STEPS:
        gpu("after_step_24")


def _step_report(env: object, step: int, action: list[float]) -> None:
    env.step(action)
    report(env, step)
    _tag_steps(step)


def _boot_env() -> tuple[object, object]:
    import config
    from envs.robocasa_env import RoboCasaEnv

    env = RoboCasaEnv.from_task(TASK, TASK_IDX)
    report(env, 0)
    gpu("after_env")
    return env, config


def _vla_loop(env: object, per_step=None) -> None:
    """VLA observe/predict/step loop (arm A core). Optional per_step(env, step) after each step."""
    from vla_client import VLAClient

    client, step = VLAClient(), 0
    for chunk_i in range(N_CHUNKS):
        images, state = env.observe()
        chunk = client.predict(images, state, PROMPT)
        if chunk_i == 0:
            gpu("after_predict_1")
        for action in chunk[:CHUNK_STEPS]:
            step += 1
            env.step(action)
            report(env, step)
            _tag_steps(step)
            if per_step is not None:
                per_step(env, step)


# --- Arm A: VLA server on GPU, no Runner / writer ---
def arm_a() -> None:
    import config

    config.SAVE_VIDEO = False
    env, _ = _boot_env()
    _vla_loop(env)
    env.close()


# --- Arm B: cv2 writer + replayed actions, no server ---
def arm_b(log_path: str) -> None:
    import config

    config.SAVE_VIDEO = True
    config.VIDEO_PATH = "/tmp/probe_b.mp4"
    flat = [a for ch in _load_chunks(log_path) for a in ch[:CHUNK_STEPS]]
    env, _ = _boot_env()
    for step, action in enumerate(flat, start=1):
        _step_report(env, step, action)
    env.close()
    import cv2

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        print(
            f"video_frame={frame_i} mean={float(np.mean(frame.astype(float))):.2f}",
            flush=True,
        )
        frame_i += 1
    cap.release()


# --- Arm C: full Runner path (server + LLM), writer off ---
class _ArmCDone(BaseException):
    pass


def arm_c() -> None:
    import config

    config.SAVE_VIDEO = False
    from envs.robocasa_env import RoboCasaEnv
    from runner import Runner
    from utils import configure_logging

    configure_logging()
    orig_init, orig_step = RoboCasaEnv._init_env, RoboCasaEnv.step

    def wrapped_init(self, task: str) -> None:
        orig_init(self, task)
        report(self, 0)
        gpu("after_env")

    def wrapped_step(self, action: list[float]) -> bool:
        out = orig_step(self, action)
        if self.steps <= TOTAL_STEPS:
            report(self, self.steps)
            _tag_steps(self.steps)
        if self.steps == TOTAL_STEPS:
            print("ARM_C_DONE", flush=True)
            raise _ArmCDone()
        return out

    RoboCasaEnv._init_env = wrapped_init
    RoboCasaEnv.step = wrapped_step
    try:
        Runner(TASK, TASK_IDX).run()
    except _ArmCDone:
        pass


# --- Arm D: import Runner tree before env, then arm A loop ---
def arm_d() -> None:
    import config

    config.SAVE_VIDEO = False
    import executor  # noqa: F401
    import plan  # noqa: F401
    import planner  # noqa: F401
    import runner  # noqa: F401
    import skills  # noqa: F401
    import utils  # noqa: F401

    env, _ = _boot_env()
    _vla_loop(env)
    env.close()


def _construct_planner(env: object) -> object:
    from planner import LLMPlanner
    from skills import skill_catalog

    return LLMPlanner(env.task_description, skill_catalog())


# --- Arm E: planner construction after env, no plan() ---
def arm_e() -> None:
    import config

    config.SAVE_VIDEO = False
    env, _ = _boot_env()
    _construct_planner(env)
    print("E: planner constructed", flush=True)
    _loaded_watch(_E_EXTRA)
    _vla_loop(env)
    env.close()


# --- Arm F: real LLM plan(), then arm A loop ---
def arm_f() -> None:
    import config

    config.SAVE_VIDEO = False
    env, _ = _boot_env()
    llm_planner = _construct_planner(env)
    images, _ = env.observe()
    plan_out = llm_planner.plan(images[0])
    print(plan_out, flush=True)
    print("F: plan done", flush=True)
    _vla_loop(env)
    env.close()


def _arm_g_post(env: object, step: int) -> None:
    env.check_subtask({"skill": "open", "target_name": "left drawer"})
    env.calc_distances("EEF", "left drawer")
    report(env, step, post=True)


# --- Arm G: executor-like GT calls after each step, no Runner ---
def arm_g() -> None:
    import config

    config.SAVE_VIDEO = False
    env, _ = _boot_env()
    _vla_loop(env, per_step=_arm_g_post)
    env.close()


# --- Arm H: import module after env, then arm A loop ---
def arm_h(mod_name: str) -> None:
    import config

    config.SAVE_VIDEO = False
    env, _ = _boot_env()
    importlib.import_module(mod_name)
    print(f"H: imported {mod_name}", flush=True)
    _vla_loop(env)
    env.close()


# --- Arm I: pre-import langchain_openai, planner after env ---
def arm_i() -> None:
    import langchain_openai  # noqa: F401

    print("I: pre-imported langchain_openai", flush=True)
    import config

    config.SAVE_VIDEO = False
    env, _ = _boot_env()
    _construct_planner(env)
    print("E: planner constructed", flush=True)
    _loaded_watch(_E_EXTRA)
    _vla_loop(env)
    env.close()


# --- Arm J: arm I + plan(), then arm A loop ---
def arm_j() -> None:
    import langchain_openai  # noqa: F401

    print("I: pre-imported langchain_openai", flush=True)
    import config

    config.SAVE_VIDEO = False
    env, _ = _boot_env()
    llm_planner = _construct_planner(env)
    images, _ = env.observe()
    plan_out = llm_planner.plan(images[0])
    print(plan_out, flush=True)
    print("J: plan done", flush=True)
    _vla_loop(env)
    env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Colab video obs bisect probe")
    p.add_argument(
        "--arm",
        required=True,
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )
    p.add_argument("--log", help="run log for arm B action replay")
    p.add_argument("--mod", help="module name for arm H post-env import")
    args = p.parse_args()
    if args.arm == "B" and not args.log:
        p.error("--log is required when --arm is B")
    if args.arm == "H" and not args.mod:
        p.error("--mod is required when --arm is H")
    gpu("start")
    _banner()
    arms = {
        "A": arm_a,
        "B": lambda: arm_b(args.log),
        "C": arm_c,
        "D": arm_d,
        "E": arm_e,
        "F": arm_f,
        "G": arm_g,
        "H": lambda: arm_h(args.mod),
        "I": arm_i,
        "J": arm_j,
    }
    arms[args.arm]()


if __name__ == "__main__":
    main()
