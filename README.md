# Longor

## Goal

Making robots to successfuly accomplish **Long-horizon** tasks via an **LLM loop**.
This will include breaking the requested task into **subtasks**, creating **callable tools** via the LLM, using **VLAs** for complex arm moves, and feeding the LLM **RGB** from sim or cameras until the top-level task is done.
**Platform-generic**—ManiSkill first, then LIBERO.
Full picture: **`SRD.md`**.

## Setup

**Layout:** create two virtualenvs **inside this repo** (same folder as `README.md`): `.venv` (app/sim/LLM) and `.octo-venv` (Octo VLA server).

**1 — Client / sim / LLM (`.venv`)**

```bash
cd /path/to/Longor
python -m venv .venv
source .venv/Scripts/activate    # Git-Bash on Windows
pip install -r requirements.txt
```

The PickCube example also needs **ManiSkill / SAPIEN** (not in `requirements.txt`); install per your machine and upstream docs. Optional hardware stacks: `requirements-optional.txt`.

**2 — Octo VLA HTTP service (`.octo-venv`)**

```bash
cd /path/to/Longor
python -m venv .octo-venv
source .octo-venv/Scripts/activate
# One-time: install JAX + FastAPI + Octo stack for services/octo_service.py (weights e.g. under ~/.cache/octo-small-1.5).
```

**Config:** create `.env` in this repo (see `.env.example` if present) and set LLM keys (e.g. `OPENAI_API_KEY` if using OpenAI) and any VLA vars you need.

## Usage

**Terminal A — VLA server**

```bash
cd /path/to/Longor
source .octo-venv/Scripts/activate
python services/octo_service.py
```

Wait until the service reports healthy on port **8765** (or the port in your `.env`).

**Terminal B — PickCube example (with VLA)**

```bash
cd /path/to/Longor
source .venv/Scripts/activate
VLA_SERVICE_URL=http://localhost:8765 python examples/maniskill_pickcube.py
```

You can also run `python examples/maniskill_pickcube.py` from `Longor/`; examples prepend `src` on `sys.path` automatically.

**Regression tests**

```bash
cd /path/to/Longor
source .venv/Scripts/activate
pytest
```

(`pytest.ini` sets `pythonpath = src`.)

---

More detail: **`SRD.md`**.
