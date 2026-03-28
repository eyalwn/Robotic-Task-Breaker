# Longor

This file lives at the **repository root** (the directory that contains `src/`, `examples/`, and `services/`). If your checkout is nested (e.g. `…/RoboCrew/Longor`), `cd` to that inner folder before running the commands below.

## Goal

Making robots to successfuly accomplish **Long-horizon** tasks via an **LLM loop**.
This will include breaking the requested task into **subtasks**, creating **callable tools** via the LLM, using **VLAs** for complex arm moves, and feeding the LLM **RGB** from sim or cameras until the top-level task is done.
**Platform-generic**—ManiSkill first, then LIBERO.
Full picture: **`SRD.md`**.

## Setup

**Layout:** create two virtualenvs **inside this repo** (same folder as `README.md`): `.venv` (app/sim/LLM) and `.octo-venv` (Octo VLA server).

**Activate a venv**

| OS | Command |
| --- | --- |
| **Linux / macOS** | `source .venv/bin/activate` or `source .octo-venv/bin/activate` |
| **Windows (Git Bash)** | `source .venv/Scripts/activate` or `source .octo-venv/Scripts/activate` |
| **Windows (cmd)** | `.venv\Scripts\activate.bat` |
| **Windows (PowerShell)** | `.venv\Scripts\Activate.ps1` |

**1 — Client / sim / LLM (`.venv`)**

```bash
cd /path/to/Longor
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# source .venv/Scripts/activate  # Windows — Git Bash
pip install -r requirements.txt
```

**ManiSkill** (required for `examples/maniskill_pickcube.py`) is **not** in `requirements.txt`. Install per the official guide: [ManiSkill — installation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html) (typically `pip install mani_skill` plus a matching **PyTorch** install). Upstream targets **Linux + NVIDIA GPU + Vulkan**; **Windows** is supported but often more fragile—keep `MANISKILL_SIM_BACKEND` / driver notes from `.env.example` when debugging.

**Environment variables**

- Copy the committed template once: `cp .env.example .env` (Git-Bash on Windows: same command).
- Edit `.env` locally. **Do not commit `.env`** — it is gitignored; only `.env.example` is shared.
- Set at least one LLM credential your chosen `LLM_MODEL` needs (e.g. `OPENAI_API_KEY` for OpenAI, or `GOOGLE_API_KEY` for Google models). Other keys in `.env.example` are safe defaults you can adjust (VLA port, ManiSkill options, etc.).

`python-dotenv` loads `.env` from the repo root when you run examples and `LLMAgent`.

**2 — Octo VLA HTTP service (`.octo-venv`)**

Use a **separate** venv: Octo/JAX versions are sensitive and differ from the app venv.

```bash
cd /path/to/Longor
python -m venv .octo-venv
source .octo-venv/bin/activate        # Linux / macOS
# source .octo-venv/Scripts/activate  # Windows — Git Bash
```

1. **Install Octo** from [octo-models/octo](https://github.com/octo-models/octo) (their README: `pip install -e .` and `pip install -r requirements.txt` inside a clone of that repo). Use a Python version they support (often 3.10).
2. **Install JAX** for your hardware ([JAX installation](https://jax.readthedocs.io/en/latest/installation.html)):
   - **Linux + NVIDIA (recommended for this project):** e.g. `pip install -U "jax[cuda12]"` or `"jax[cuda13]"` matching your driver/CUDA stack (see JAX docs; CUDA wheels are **Linux x86_64**).
   - **CPU-only / Windows:** `pip install -U jax` (CPU). NVIDIA GPU wheels for JAX are **not** available for native Windows; use CPU JAX or WSL2/Linux for GPU Octo.
3. **HTTP shim deps:** from this repo’s root, with `.octo-venv` active:  
   `pip install -r services/requirements-octo.txt`
4. **Weights:** `services/octo_service.py` loads a **local directory** `VLA_MODEL_DIR` (default `~/.cache/octo-small-1.5`). Download or sync the **octo-small** checkpoint from [Hugging Face — rail-berkeley](https://huggingface.co/rail-berkeley) into that path so the directory exists and matches what `OctoModel.load_pretrained` expects for a filesystem path.

If Octo’s pinned JAX and a GPU JAX build conflict, prefer adjusting JAX **after** Octo’s base install per Octo/JAX issue trackers, or use the versions Octo documents for GPU.

## Usage

**Terminal A — VLA server**

```bash
cd /path/to/Longor
source .octo-venv/bin/activate        # Linux / macOS
# source .octo-venv/Scripts/activate   # Windows — Git Bash
python services/octo_service.py
```

Wait until the service reports healthy on the port from your `.env` (default **8765**, `VLA_PORT`).

**Terminal B — PickCube example (with VLA)**

```bash
cd /path/to/Longor
source .venv/bin/activate             # Linux / macOS
# source .venv/Scripts/activate       # Windows — Git Bash
python examples/maniskill_pickcube.py
```

The example talks to Octo at `http://localhost:${VLA_PORT}` (set `VLA_PORT` in `.env` if you changed the server port).

You can also run `python examples/maniskill_pickcube.py` from `Longor/`; examples prepend `src` on `sys.path` automatically.

**Regression tests**

```bash
cd /path/to/Longor
source .venv/bin/activate             # Linux / macOS
# source .venv/Scripts/activate       # Windows — Git Bash
pytest
```

(`pytest.ini` sets `pythonpath = src`.)

**Linux + GPU checklist (M1)**

- NVIDIA driver + Vulkan usable for ManiSkill (see ManiSkill docs).
- In `.env`, try clearing **`MANISKILL_SIM_BACKEND`** or set a GPU backend only if ManiSkill docs recommend it for your build; **`MANISKILL_SIM_BACKEND=cpu`** stays the safe default for Windows and for headless CPU-only Linux.
- **VLA axis remap/signs** in `.env.example` were tuned for **Windows CPU** PickCube; on Linux/GPU you may need different values—treat them as calibration, not universal constants.

---

More detail: **`SRD.md`**.
