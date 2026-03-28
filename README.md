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

The PickCube example also needs **ManiSkill / SAPIEN** (not in `requirements.txt`); install per your machine and upstream docs.

**Environment variables**

- Copy the committed template once: `cp .env.example .env` (Git-Bash on Windows: same command).
- Edit `.env` locally. **Do not commit `.env`** — it is gitignored; only `.env.example` is shared.
- Set at least one LLM credential your chosen `LLM_MODEL` needs (e.g. `OPENAI_API_KEY` for OpenAI, or `GOOGLE_API_KEY` for Google models). Other keys in `.env.example` are safe defaults you can adjust (VLA port, ManiSkill options, etc.).

`python-dotenv` loads `.env` from the repo root when you run examples and `LLMAgent`.

**2 — Octo VLA HTTP service (`.octo-venv`)**

```bash
cd /path/to/Longor
python -m venv .octo-venv
source .octo-venv/Scripts/activate
# One-time: install JAX + FastAPI + Octo stack for services/octo_service.py (weights e.g. under ~/.cache/octo-small-1.5).
```

## Usage

**Terminal A — VLA server**

```bash
cd /path/to/Longor
source .octo-venv/Scripts/activate
python services/octo_service.py
```

Wait until the service reports healthy on the port from your `.env` (default **8765**, `VLA_PORT`).

**Terminal B — PickCube example (with VLA)**

```bash
cd /path/to/Longor
source .venv/Scripts/activate
python examples/maniskill_pickcube.py
```

The example talks to Octo at `http://localhost:${VLA_PORT}` (set `VLA_PORT` in `.env` if you changed the server port).

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
