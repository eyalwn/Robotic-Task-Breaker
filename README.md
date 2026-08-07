# Longor

Long-horizon robot tasks: an **LLM** emits a structured plan, a stateless
**executor** walks one plan at a time, a **VLA** policy does the arm
manipulation, and **RGB** feedback checkpoints re-invoke planning with a runner
snapshot (planner keeps progress history and may emit a new plan or a sentinel
STOP plan).

Platform: **LIBERO on Linux + NVIDIA** (Colab T4 is sufficient). Architecture and
contracts: `SRD.md`.

Ownership split (runtime):
- `Runner`: composition root — owns `LiberoEnv` and `SkillRegistry`; builds `LLMPlanner`; loop is `observe -> plan(snapshot) -> executor.run(plan)`, then `observe + plan(snapshot, execution_sum)` on feedback; checks sentinel STOP before execute.
- `LLMPlanner`: `plan(snapshot, execution_sum?) -> Plan`; owns bounded progress history across calls; STOP is a sentinel return literal `"STOP: <reason>"`; model/retries from `config`.
- `Executor`: stateless `run(plan) -> ExecutionSum`; owns VLA/env execution and no cross-run memory.

## Setup

Setup/runbook details live in this README. Notebook launchers under `colab/`
are the executable setup path for Colab flows.

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

On Colab, use `colab/libero_probe.ipynb` as the setup launcher — it installs
deps, starts `services/smolvla_service.py`, and runs the probe.

## Tests

With `.venv` active:

```bash
pytest
```

Unit tests are hermetic: no live sim, no network. They cover the plan AST,
executor (control flow + expression eval), VLA client wire contract, and
planner JSON extraction/retry.

Opt-in live planner check (non-hermetic, billed OpenAI calls; Windows-ok; no
LIBERO). Needs `OPENAI_API_KEY` in `.env` / environment:

```bash
pytest -m integration tests/test_planner_live_llm.py -q
```

Before running, download a real LIBERO task-start snapshot and place it at:
`tests/fixtures/libero_object_task0_start.jpg`.

The test forces `MAX_PLAN_RETRIES=1` across a few independent `plan()` calls.
It skips unless you run with `-m integration`, set the API key, and provide the
snapshot file.
