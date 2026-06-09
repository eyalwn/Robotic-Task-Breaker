# Longor

Note: This file lives at the **repository root** (the directory that contains `src/`, `examples/`, and `services/`). If your checkout is nested (e.g. `…/RoboCrew/Longor`), `cd` to that inner folder first.

**Supported dev hosts:** **Linux** and **Windows (Git Bash)** only.

## Goal

Long-horizon robot tasks via an **LLM loop**, **VLAs** for arm moves, **RGB** feedback from sim or cameras. **ManiSkill** first, then LIBERO (see `SRD.md`).

## Setup

Create **two** virtualenvs in this directory: `.venv` (app / sim / LLM) and `.vla-venv` (VLA HTTP server only). You may use `python -m pip` via the venv's interpreter so you do not need to activate while installing.

### 1 — Environment file

1. `cp .env.example .env`
2. Set secrets (e.g. `OPENAI_API_KEY`). Set `**VLA_POLICY`** to an id implemented in `services/run_vla_server.py` (e.g. `octo-small-1.5`). Octo loads `**rail-berkeley/octo-small-1.5**` from Hugging Face on first service start (see `services/octo_service.py`).

**Never commit `.env` (gitignored).**

### 2 — App venv (`.venv`)

From the repo root:

**Linux (with CUDA)**

GPU PyTorch for ManiSkill: **do not** run `pip install -r requirements-maniskill.txt` here; that file pins **CPU** `torch`/`torchvision`. Install **CUDA** PyTorch first (use the `pip` command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for your CUDA version), then ManiSkill:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
# Then run the PyTorch+CUDA install line from pytorch.org (not shown here — it varies by CUDA).
.venv/bin/python -m pip install "mani-skill>=3.0.0b20"
```

**Windows (Git Bash)**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -U pip
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -r requirements-maniskill.txt
```

**Note:** Rationale for `requirements-maniskill.txt` vs CUDA is in the header of `requirements-maniskill.txt`.
**Note:** keep `MANISKILL_ENV=PickCube-v1` in `.env` for ManiSkill examples.

#### LIBERO (Linux only)

LIBERO requires **Linux** + MuJoCo. `requirements-libero.txt` is self-contained (`-r requirements.txt` included), so it works as a standalone install or on top of an existing `.venv`:

```bash
.venv/bin/python -m pip install -r requirements-libero.txt
export MUJOCO_GL=egl  # headless rendering (add to .bashrc)
```

Set `BACKEND=libero` in `.env`.

### 3 — VLA venv

**One stack per venv.** If pip conflicts appear when switching policies, delete `.vla-venv` and recreate. Set `VLA_POLICY` in `.env` to match the installed stack.

Pick **one** of the VLA stacks below.

---

#### Octo `octo-small-1.5`

##### Windows (Git Bash)

Use **Python 3.11** (`py -3.11`).

```bash
py -3.11 -m venv .vla-venv
.vla-venv/Scripts/python -m pip install -U pip
git clone https://github.com/octo-models/octo.git ~/octo
.vla-venv/Scripts/python -m pip install -r services/requirements-vla-octo-small-1.5-windows.txt
.vla-venv/Scripts/python -m pip install -e ~/octo --no-deps
```

First service start downloads weights from [rail-berkeley/octo-small-1.5](https://huggingface.co/rail-berkeley/octo-small-1.5) (needs network).

##### Linux (with CUDA)

```bash
python3 -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
.vla-venv/bin/python -m pip install -U "jax[cuda12]"
# or jax[cuda13] — https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu
git clone https://github.com/octo-models/octo.git ~/octo
.vla-venv/bin/python -m pip install -e ~/octo
.vla-venv/bin/python -m pip install -r ~/octo/requirements.txt
.vla-venv/bin/python -m pip install -r services/requirements-vla-octo-small-1.5-linux.txt
```

Same Hugging Face weights as on Windows. If Octo's pins fight your JAX install, recreate the venv and follow Octo's README / issues.

Set `VLA_POLICY=octo-small-1.5` in `.env`.

---

#### Pi0.5 (OpenPI) — Linux + NVIDIA GPU only

Requires **NVIDIA GPU ≥ 8 GB VRAM**. Not supported on Windows.

```bash
python3 -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git ~/openpi
.vla-venv/bin/python -m pip install -e ~/openpi
.vla-venv/bin/python -m pip install -r services/requirements-vla-pi05-linux.txt
```

Checkpoints download automatically on first service start from `gs://openpi-assets` (cached in `~/.cache/openpi`). If OpenPI's deps conflict, see [OpenPI installation](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#installation) for `uv`-based setup.

Set `VLA_POLICY=pi05` in `.env`.

## Usage

**Terminal A — VLA server**
Linux (with CUDA):

```bash
cd /path/to/Longor
source .vla-venv/bin/activate
python services/run_vla_server.py
```

Windows (Git Bash):

```bash
cd /path/to/Longor
source .vla-venv/Scripts/activate
python services/run_vla_server.py
```

`VLA_POLICY` in `.env` selects the implementation (e.g. `octo-small-1.5`). Wait until `GET /health` on `VLA_PORT` (default **8765**) succeeds.

**Terminal B — sim example**

*ManiSkill PickCube (Linux or Windows):*

```bash
cd /path/to/Longor
source .venv/bin/activate
python examples/maniskill_pickcube.py
```

Windows (Git Bash):

```bash
cd /path/to/Longor
source .venv/Scripts/activate
python examples/maniskill_pickcube.py
```

The client uses `http://localhost:${VLA_PORT}`. Examples add `src` to `sys.path` automatically.

*LIBERO-Object (Linux only):*

```bash
cd /path/to/Longor
source .venv/bin/activate
export MUJOCO_GL=egl
python examples/libero_object.py
```

## Tests

with '.venv' activated, run:

```bash
pytest
```

`pytest.ini` sets `pythonpath = src`.

**Integration tests:**
`pytest -m integration` (needs a running VLA server).

**Linux + GPU notes**

- ManiSkill wants a working **Vulkan** stack for typical setups (see ManiSkill docs).
- `MANISKILL_SIM_BACKEND` defaults to `cpu` on Windows and unset (GPU) on Linux; override in `.env` only if needed.
- `**VLA_AXIS_REMAP` / `VLA_AXIS_SIGNS**` in `.env.example` were tuned for **Windows CPU** PickCube; re-tune on Linux/GPU if motion looks wrong.

---

More detail: `SRD.md`.
