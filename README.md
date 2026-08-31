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



### 3 - Policy venv

```bash
python3 -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
.vla-venv/bin/python -m pip install -r deps/requirements-vla-smolvla-linux.txt
.vla-venv/bin/python -m pip install -e .
```

Default checkpoint is `HuggingFaceVLA/smolvla_libero`; override with
`SMOLVLA_CHECKPOINT` in `.env` if needed.

## Usage

**Terminal A - policy service.** Wait for it to report that it is listening
(startup includes a checkpoint download on first run).

```bash
source .vla-venv/bin/activate
python services/smolvla_service.py
```

**Terminal B - the run.**

```bash
source .venv/bin/activate
python examples/long_horizon_llm_libero.py
```

The client talks to `http://127.0.0.1:8765`.

Set `SAVE_VIDEO=1` in `.env` to write `run.mp4`.

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