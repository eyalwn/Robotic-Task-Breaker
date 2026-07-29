# Longor

Long-horizon robot tasks: an **LLM** emits a structured plan, an **executor**
walks it, a **VLA** policy does the arm manipulation, and **RGB** feedback routes
the run to continue, stop, or replan.

Platform: **LIBERO on Linux + NVIDIA** (Colab T4 is sufficient). Architecture and
contracts: `SRD.md`.

## Setup

Two virtualenvs in the repo root: `.venv` for the app and simulation, `.vla-venv`
for the policy server. They must be separate — the policy stack and the sim stack
pin incompatible dependencies.

### 1 — Environment file

```bash
cp .env.example .env
```

Fill in `OPENAI_API_KEY`. The other three variables have working defaults. These
four are the only environment variables the code reads; everything else is a
constant in `src/config.py`.

**Never commit `.env`** (gitignored).

### 2 — App venv

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r deps/requirements-libero.txt
.venv/bin/python -m pip install -e .
```

`deps/requirements-libero.txt` is self-contained (it includes
`deps/requirements.txt`). The final `pip install -e .` is required — without it
the `src/` modules are not importable.

Headless rendering needs these exported in every shell that runs the sim (add to
`.bashrc`):

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 3 — Policy venv

```bash
python3 -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
.vla-venv/bin/python -m pip install -r deps/requirements-vla-smolvla-linux.txt
.vla-venv/bin/python -m pip install -e .
```

Default checkpoint is `HuggingFaceVLA/smolvla_libero`; override with
`SMOLVLA_CHECKPOINT` in `.env` if needed.

## Usage

**Terminal A — policy service.** Wait for it to report that it is listening
(startup includes a checkpoint download on first run).

```bash
source .vla-venv/bin/activate
python services/smolvla_service.py
```

**Terminal B — the run.**

```bash
source .venv/bin/activate
python examples/long_horizon_llm_libero.py
```

The client talks to `http://127.0.0.1:8765`.

## Validating the VLA to environment path

Before debugging anything in the planner or executor, confirm the policy can
solve a task with nothing else in the loop:

```bash
source .venv/bin/activate
python validation/libero_probe.py                    # libero_object, task 0
python validation/libero_probe.py --suite libero_10 --task 0
```

It prints EE displacement, closest approach to each task object, gripper travel,
and mean action magnitude per dimension, then exits 0 only if the task was
solved. A failure here means the environment, the service, or the checkpoint —
never the orchestration.

Set `SAVE_VIDEO=1` in `.env` to get `run.mp4` showing exactly what the policy saw.

## Tests

With `.venv` active:

```bash
pytest
```

Unit tests are hermetic: no live sim, no network. `pytest -m integration` runs
the tests that need a live policy service.
