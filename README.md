# Longor

Longor is a thin software layer that lets an LLM break a long-horizon robot
task into short steps a VLA can execute.

It is platform-generic: switch LLM, VLA, or simulator with a venv plus
constants in `src/config.py`. One of each per process. Contracts: `SRD.md`.

Default stack: **VLAbench** + **SmolVLA** (`lerobot/smolvla_vlabench`),
Linux + NVIDIA/CUDA.

## Setup

Two envs are required in repo root:

- `.venv` for app + simulation
- `.vla-venv` for the policy server

### 1 - Environment file

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` (do not commit this file).

### 2 - App venv (VLAbench)

Python 3.10. Run each command from the repo root:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -U pip
git clone https://github.com/OpenMOSS/VLABench .vendor/VLABench
git -C .vendor/VLABench checkout cf588fe60c0c7282174fe979f5913170cfe69017
git clone https://github.com/motion-planning/rrt-algorithms .vendor/rrt-algorithms
git -C .vendor/rrt-algorithms checkout e51d95ee489a225220d6ae2a764c4111f6ba7d85
touch .vendor/rrt-algorithms/rrt_algorithms/rrt/__init__.py
touch .vendor/rrt-algorithms/rrt_algorithms/search_space/__init__.py
touch .vendor/rrt-algorithms/rrt_algorithms/utilities/__init__.py
.venv/bin/python -m pip install -r deps/requirements-vlabench.txt
.venv/bin/python -m pip install -e .vendor/rrt-algorithms
.venv/bin/python -m pip install -e .vendor/VLABench
.venv/bin/python -m pip install -e .
.venv/bin/python .vendor/VLABench/scripts/download_assets.py
```

For headless sim shells:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

LIBERO remains a non-default backend (`SIM_ENV="libero"` plus
`deps/requirements-libero.txt`).

### 3 - Policy venv (default: SmolVLA / VLAbench)

```bash
python3 -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
.vla-venv/bin/python -m pip install -r deps/requirements-vla-smolvla-linux.txt
.vla-venv/bin/python -m pip install -e .
```

Default checkpoint is `lerobot/smolvla_vlabench`. Override with
`SMOLVLA_CHECKPOINT` when needed.

#### Optional: π0.5 / OpenPI

Linux + NVIDIA. Clone OpenPI as a sibling of this repo. The `lerobot @ git+...`
pin must be in the same `pip install -e ../openpi` command — pip ignores
OpenPI's `[tool.uv.sources]`, and current PyPI `lerobot` needs `numpy>=2`
while OpenPI pins `numpy<2`. Prefer a fresh `.vla-venv` if switching backends.

```bash
python3 -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi ../openpi
.vla-venv/bin/python -m pip install -e ../openpi \
  "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
.vla-venv/bin/python -m pip install -r deps/requirements-vla-pi05-linux.txt
.vla-venv/bin/python -m pip install -e .
```

## Usage

**Terminal A - policy service.** Wait for it to report that it is listening
(startup includes a checkpoint download on first run).

```bash
source .vla-venv/bin/activate
python services/run_vla_server.py          # default: smolvla
# python services/run_vla_server.py --policy pi05
```

**Terminal B - the run.**

```bash
source .venv/bin/activate
python examples/run_vlabench_task.py
# python examples/run_libero_task.py   # requires SIM_ENV=libero and LIBERO deps
```

The client talks to `http://127.0.0.1:8765` and is backend-agnostic.

Set `SAVE_VIDEO=1` in `.env` to write `run.mp4`.

Colab launcher: `colab/run_vlabench_task.ipynb` (runs `examples/run_vlabench_task.py`).

## Tests

With `.venv` active:

```bash
pytest
```

Unit tests are hermetic: no live sim, no network. They cover executor step
flow, runner stop/verdict flow, plan parsing, VLA client wire contract, and
planner JSON extraction/retry.

Opt-in live planner check (non-hermetic, billed OpenAI calls; Windows-ok; no
sim). Needs `OPENAI_API_KEY` in `.env` / environment:

```bash
pytest -m integration tests/test_planner_live_llm.py -q
```
