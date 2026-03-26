# Longor

## Goal

Making robots to successfuly accomplish **Long-horizon** tasks via an **LLM loop**.
This will include breaking the requested task into **subtasks**, creating **callable tools** via the LLM, using **VLAs** for complex arm moves, and feeding the LLM **RGB** from sim or cameras until the top-level task is done.
**Platform-generic**—ManiSkill first, then LIBERO.
Full picture: **`SRD.md`**.

## Setup

**Layout:** keep two virtualenvs **next to** this repo (parent of `longor/`), e.g. `robotics/.venv` and `robotics/.octo-venv`.

**1 — Client / sim / LLM (`.venv`)**

```bash
cd ..                    # parent of longor/
python -m venv .venv
source .venv/Scripts/activate    # Git-Bash on Windows
pip install -r longor/requirements.txt
```

The PickCube example also needs **ManiSkill / SAPIEN** (not in `requirements.txt`); install per your machine and upstream docs. Optional hardware stacks: `longor/requirements-optional.txt`.

**2 — Octo VLA HTTP service (`.octo-venv`)**

```bash
cd ..                    # same parent as above
python -m venv .octo-venv
source .octo-venv/Scripts/activate
# One-time: install JAX + FastAPI + Octo stack for services/octo_service.py (weights e.g. under ~/.cache/octo-small-1.5).
```

**Config:** create a file `longor/.env` and set LLM keys (e.g. `OPENAI_API_KEY` if using OpenAI) and any VLA vars you need.

## Usage

**Terminal A — VLA server**

```bash
# at parent of longor/
source .octo-venv/Scripts/activate
python longor/services/octo_service.py
```

Wait until the service reports healthy on port **8765** (or the port in your `.env`).

**Terminal B — PickCube example (with VLA)**

```bash
# at parent of longor/
source .venv/Scripts/activate
python examples/maniskill_pickcube.py
```

**Regression tests**

```bash
# at parent of longor/
source .venv/Scripts/activate
pytest
```

(`pytest.ini` sets `pythonpath = src`.)

---

More detail: **`SRD.md`**.
