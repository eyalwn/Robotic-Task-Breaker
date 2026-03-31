# Longor

Note: This file lives at the **repository root** (the directory that contains `src/`, `examples/`, and `services/`). If your checkout is nested (e.g. `…/RoboCrew/Longor`), `cd` to that inner folder first.

**Supported dev hosts:** **Linux** and **Windows (Git Bash)** only.

## Goal

Long-horizon robot tasks via an **LLM loop**, **VLAs** for arm moves, **RGB** feedback from sim or cameras. **ManiSkill** first, then LIBERO (see `SRD.md`).

## Setup

Create **two** virtualenvs in this directory: `.venv` (app / sim / LLM) and `.vla-venv` (VLA HTTP server only). You may use `python -m pip` via the venv’s interpreter so you do not need to activate while installing.

### 1 — Environment file

1. `cp .env.example .env`
2. Set secrets (e.g. `OPENAI_API_KEY`) and `VLA_POLICY` (e.g. `octo-small-1.5` for the default stack).
3. Never commit `.env` (gitignored).

### 2 — App venv (`.venv`)

From the repo root:

**Linux (with CUDA)**

GPU PyTorch for ManiSkill: **do not** run `pip install -r requirements-sim.txt` here; that file pins **CPU** `torch`/`torchvision`. Install **CUDA** PyTorch first (use the `pip` command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for your CUDA version), then ManiSkill:

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
.venv/Scripts/python -m pip install -r requirements-sim.txt
```

**Note:** Rationale for `requirements-sim.txt` vs CUDA is in the header of `requirements-sim.txt`.
**Note:** keep `MANISKILL_ENV=PickCube-v1` in `.env` (until M3).

### 3 — VLA venv (`.vla-venv`)

**Flow (any policy):**
Note: the venv paths in this section refer to Linux. For Windows, replace every `.vla-venv/bin/python` command with: `.vla-venv/Scripts/python`.

1. make sure that the value in `VLA_POLICY` is listed in `services/run_vla_server.py`; adding a new policy means a new `requirements-vla-*.txt`, a server module, and a registry entry.
2. Create `.vla-venv` and upgrade `pip`.
```bash
python -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
```
**Note:** for `octo-small-1.5` use Python 3.11 or lower (due to the dependency `tensorflow_text` - tends to fail on Windows with Python **3.12+**).
3. Install JAX the way your OS needs. for Linux:
```bash
.vla-venv/bin/python -m pip install -U "jax[cuda12]"
# (`jax[cuda13]` is fine if that matches your driver — see [JAX — NVIDIA GPU](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu))
```
for Windows (JAX supports CPU only):
```bash
.vla-venv/Scripts/python -m pip install -U jax
```
4. Install the **policy’s upstream Python package** (clone + editable install — see the Octo example below; other policies follow their upstream docs).
For Octo for example, the commands will be:
```bash
git clone https://github.com/octo-models/octo.git ~/octo
.vla-venv/bin/python -m pip install -e ~/octo
```
5. Install Longor’s deps requirements for that policy (JAX + HTTP shim, etc.):
```bash
.vla-venv/bin/python -m pip install -r ~/octo/requirements.txt
.vla-venv/bin/python -m pip install -r services/requirements-vla-octo-small-1.5.txt  # replace with your policy id
```
6. Download the model's weights into `VLA_MODEL_DIR` (paths differ per policy; see `.env.example` and the policy’s upstream docs).

**Note:** Use one venv per policy stack. If you switch `VLA_POLICY` and pip conflicts appear, delete `.vla-venv`, recreate it, and reinstall.

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

**Terminal B — PickCube example**
Linux (with CUDA):

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
- In `.env`, **`MANISKILL_SIM_BACKEND=cpu`** is the safe default (especially on Windows). On Linux + GPU you may clear it or set a GPU backend per ManiSkill docs.
- **`VLA_AXIS_REMAP` / `VLA_AXIS_SIGNS`** in `.env.example` were tuned for **Windows CPU** PickCube; re-tune on Linux/GPU if motion looks wrong.

---

More detail: `SRD.md`.