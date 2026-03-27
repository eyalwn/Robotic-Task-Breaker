# Longor

## Goal

Making robots to successfuly accomplish **Long-horizon** tasks via an **LLM loop**.
This will include breaking the requested task into **subtasks**, creating **callable tools** via the LLM, using **VLAs** for complex arm moves, and feeding the LLM **RGB** from sim or cameras until the top-level task is done.
**Platform-generic**—ManiSkill first, then LIBERO.
Full picture: **`SRD.md`**.

## Setup

**Layout:** keep two virtualenvs **next to** this repo (parent of `Longor/`), e.g. `robotics/.venv` and `robotics/.octo-venv`.

**1 — Client / sim / LLM (`.venv`)**

```bash
cd ..                    # parent of Longor/
python -m venv .venv
source .venv/Scripts/activate    # Git-Bash on Windows
pip install -r Longor/requirements.txt
```

The PickCube example also needs **ManiSkill / SAPIEN** (not in `requirements.txt`); install per your machine and upstream docs. Optional hardware stacks: `Longor/requirements-optional.txt`.

**2 — Octo VLA HTTP service (`.octo-venv`)**

```bash
cd ..                    # same parent as above
python -m venv .octo-venv
source .octo-venv/Scripts/activate
# One-time: install JAX + FastAPI + Octo stack for services/octo_service.py (weights e.g. under ~/.cache/octo-small-1.5).
```

**Config:** create a file `Longor/.env` and set LLM keys (e.g. `OPENAI_API_KEY` if using OpenAI) and any VLA vars you need.

## Usage

**Terminal A — VLA server**

```bash
# at parent of Longor/
source .octo-venv/Scripts/activate
python Longor/services/octo_service.py
```

Wait until the service reports healthy on port **8765** (or the port in your `.env`).

**Terminal B — PickCube example (with VLA)**

```bash
# at parent of Longor/
source .venv/Scripts/activate
cd Longor
VLA_SERVICE_URL=http://localhost:8765 python examples/maniskill_pickcube.py
```

From the same parent directory you can also run `python Longor/examples/maniskill_pickcube.py` (no `PYTHONPATH`); examples prepend `Longor/src` automatically.

**Regression tests**

```bash
# at parent of Longor/
source .venv/Scripts/activate
cd Longor
pytest
```

(`pytest.ini` sets `pythonpath = src`.)

---

More detail: **`SRD.md`**.
