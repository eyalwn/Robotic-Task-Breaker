# Longor

Longor is a modular long-horizon robot agent for LIBERO that combines free-language planning with vision-language-action (VLA) control.

It is designed to be practical for research and demos:

- clear planner -> executor -> policy boundaries,
- simple JSON-plan interface from the LLM,
- runtime for Linux + NVIDIA/CUDA, with cross-platform development/testing workflows.

The goal is straightforward: turn high-level intent into grounded multi-step
robot behavior with minimal glue code and predictable execution.

It does three things:

- asks an **LLM** for a step-by-step plan in JSON,
- calls a **VLA server** to produce robot actions,
- executes the plan until done (or STOP).



## Setup

Two envs are required in repo root:

- `.venv` for app + simulation
- `.vla-venv` for the policy server



### 1 - Environment file

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` (do not commit this file).

### 2 - App venv

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r deps/requirements-libero.txt
.venv/bin/python -m pip install -e .
```

`deps/requirements-libero.txt` is self-contained (it includes
`deps/requirements.txt`).

For headless sim shells:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```



### 3 - Policy venv (default: π0.5 / OpenPI)

Default backend is **π0.5** (`pi05_libero`). Linux + NVIDIA only.

```bash
python3 -m venv .vla-venv
source .vla-venv/bin/activate
# Install OpenPI first (clone with submodules; follow upstream README):
#   git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi
#   cd openpi && uv sync   # or pip install -e . per upstream docs
pip install -r deps/requirements-vla-pi05-linux.txt
pip install -e .
```

Default OpenPI train config is `pi05_libero` with checkpoint
`gs://openpi-assets/checkpoints/pi05_libero/`. Override with service-local
`OPENPI_CONFIG` / `OPENPI_CHECKPOINT` when needed.

#### Fallback: SmolVLA

```bash
# Prefer a fresh .vla-venv if switching backends (dep conflicts are common).
python3 -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
.vla-venv/bin/python -m pip install -r deps/requirements-vla-smolvla-linux.txt
.vla-venv/bin/python -m pip install -e .
```

Default checkpoint is `HuggingFaceVLA/smolvla_libero`; override with
`SMOLVLA_CHECKPOINT` when needed.

## Usage

**Terminal A - policy service.** Wait for it to report that it is listening
(startup includes a checkpoint download on first run).

```bash
source .vla-venv/bin/activate
python services/run_vla_server.py          # default: pi05
# python services/run_vla_server.py --policy smolvla   # fallback
```

**Terminal B - the run.**

```bash
source .venv/bin/activate
python examples/run_libero_task.py
```

The client talks to `http://127.0.0.1:8765` and is backend-agnostic.

Set `SAVE_VIDEO=1` in `.env` to write `run.mp4`.

## Evaluation note

Local M2 (monolithic SmolVLA baseline campaign) is skipped for cost. M4/M5
compare Longor results against official OpenPI π0.5 LIBERO published rates
under matched conditions. See `SRD.md` Evaluation Protocol and
`validation/libero_published_baseline.py`.

## Tests

With `.venv` active:

```bash
pytest
```

Unit tests are hermetic: no live sim, no network. They cover executor step
flow, runner stop/verdict flow, plan parsing, VLA client wire contract, and
planner JSON extraction/retry.

Opt-in live planner check (non-hermetic, billed OpenAI calls; Windows-ok; no
LIBERO). Needs `OPENAI_API_KEY` in `.env` / environment:

```bash
pytest -m integration tests/test_planner_live_llm.py -q
```

Before running, download a real LIBERO task-start snapshot and place it at:
`tests/fixtures/libero_object_task0_start.jpg`.