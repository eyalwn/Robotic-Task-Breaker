# Longor

## Goal

Run an **LLM-driven embodied agent** in **ManiSkill** simulation on **Windows (Git-Bash)** without robot hardware, optionally using a real **VLA over HTTP** (Octo `octo-small-1.5`) instead of a stub policy. Architecture, milestones, and API contracts live in **`SRD.md`** (oriented at maintainers/agents).

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

**Config:** copy `longor/.env.example` → `longor/.env` and set LLM keys (e.g. `OPENAI_API_KEY` if using OpenAI) and any VLA vars you need.

## Usage

**Terminal A — VLA server**

```bash
cd ..                    # parent of longor/
source .octo-venv/Scripts/activate
python longor/services/octo_service.py
```

Wait until the service reports healthy on port **8765** (or the port in your `.env`).

**Terminal B — PickCube example (with VLA)**

```bash
cd ..                    # parent of longor/
source .venv/Scripts/activate
cd longor
VLA_SERVICE_URL=http://localhost:8765 PYTHONPATH=src python examples/maniskill_pickcube.py
```

Omit `VLA_SERVICE_URL` to use the **stub** policy path (no Octo process).

**Regression tests**

```bash
cd longor
source ../.venv/Scripts/activate
pytest
```

(`pytest.ini` sets `pythonpath = src`.)

---

More detail: **`SRD.md`**.
