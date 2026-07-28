# Longor

## Goal

Long-horizon robot tasks via an **LLM loop**, **VLAs** for arm moves, **RGB** feedback from sim or cameras.

## Setup

Create **two** virtualenvs in this directory: `.venv` (app / sim / LLM) and `.vla-venv` (VLA HTTP server only). You may use `python -m pip` via the venv's interpreter so you do not need to activate while installing.

### 1 — Environment file

1. `cp .env.example .env`
2. Set secrets (e.g. `OPENAI_API_KEY`). Set `**VLA_POLICY`** to an id implemented in `services/run_vla_server.py` (for LIBERO use `smolvla`; for ManiSkill use `octo-small-1.5`).

**Never commit `.env` (gitignored).**

### 2 — App venv (`.venv`)

From the repo root:

**Linux (with CUDA)**

GPU PyTorch for ManiSkill: **do not** run `pip install -r deps/requirements-maniskill.txt` here; that file pins **CPU** `torch`/`torchvision`. Install **CUDA** PyTorch first (use the `pip` command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for your CUDA version), then ManiSkill:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r deps/requirements.txt
.venv/bin/python -m pip install -e .
# Then run the PyTorch+CUDA install line from pytorch.org (not shown here — it varies by CUDA).
.venv/bin/python -m pip install "mani-skill>=3.0.0b20"
```

**Windows (Git Bash)**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -U pip
.venv/Scripts/python -m pip install -r deps/requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pip install -r deps/requirements-maniskill.txt
```

**Note:** Rationale for `deps/requirements-maniskill.txt` vs CUDA is in the header of that file.
**Note:** ManiSkill examples pass `env_id` explicitly (e.g. `PickCube-v1`); `MANISKILL_ENV` is only used by `maniskill_rgbd_diag`.

#### LIBERO (Linux only)

LIBERO requires **Linux** + MuJoCo. `deps/requirements-libero.txt` is self-contained (`-r requirements.txt` included), so it works as a standalone install or on top of an existing `.venv`:

```bash
.venv/bin/python -m pip install -r deps/requirements-libero.txt
export MUJOCO_GL=egl  # headless rendering (add to .bashrc)
```

Set `BACKEND=libero` in `.env`.

### 3 — VLA venv

**One stack per venv.** If pip conflicts appear when switching policies, delete `.vla-venv` and recreate. Set `VLA_POLICY` in `.env` to match the installed stack.

Every VLA stack **must** end with `pip install -e .` from the repo root so `services/run_vla_server.py` can import shared helpers (`utils`).

Pick **one** of the VLA stacks below.

---

#### SmolVLA `smolvla` (primary LIBERO VLA on Linux/Colab)

Requires **Linux + NVIDIA GPU** (Colab T4 is sufficient).

```bash
python3 -m venv .vla-venv
.vla-venv/bin/python -m pip install -U pip
.vla-venv/bin/python -m pip install -r deps/requirements-vla-smolvla-linux.txt
.vla-venv/bin/python -m pip install -e .
```

Default checkpoint is `HuggingFaceVLA/smolvla_libero`. Override with `SMOLVLA_CHECKPOINT` in `.env` if needed.

Set `VLA_POLICY=smolvla` in `.env`.

---

#### Octo `octo-small-1.5`

##### Windows (Git Bash)

Use **Python 3.11** (`py -3.11`).

```bash
py -3.11 -m venv .vla-venv
.vla-venv/Scripts/python -m pip install -U pip
git clone https://github.com/octo-models/octo.git ~/octo
.vla-venv/Scripts/python -m pip install -r deps/requirements-vla-octo-small-1.5-windows.txt
.vla-venv/Scripts/python -m pip install -e ~/octo --no-deps
.vla-venv/Scripts/python -m pip install -e .
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
.vla-venv/bin/python -m pip install -r deps/requirements-vla-octo-small-1.5-linux.txt
.vla-venv/bin/python -m pip install -e .
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
.vla-venv/bin/python -m pip install -r deps/requirements-vla-pi05-linux.txt
.vla-venv/bin/python -m pip install -e .
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

`VLA_POLICY` in `.env` selects the implementation (e.g. `smolvla`, `octo-small-1.5`, `pi05`). Wait until `GET /health` on `VLA_PORT` (default **8765**) succeeds.

**Terminal B — example**

*ManiSkill PickCube (Linux or Windows):*

```bash
cd /path/to/Longor
source .venv/bin/activate   # or .venv/Scripts/activate on Windows
python examples/maniskill_pickcube.py
```

*LIBERO — LLM plan (Linux only, requires `OPENAI_API_KEY` in `.env`):*

```bash
cd /path/to/Longor
source .venv/bin/activate
export VLA_POLICY=smolvla
export MUJOCO_GL=egl
python examples/long_horizon_llm_libero.py
```

*XLeRobot hardware demos* live under `examples/0_xlerobot_*.py` through `examples/3_xlerobot_*.py`.

The client uses `http://localhost:${VLA_PORT}`. After `pip install -e .`, examples import `core` / `sim` / `robots` directly. LIBERO VLA smoke checks live in `tests/test_libero_integration.py` (`pytest -m integration`).

## Tests

with '.venv' activated, run:

```bash
pytest
```

`pyproject.toml` `[tool.pytest.ini_options]` sets `pythonpath = src`.

**Integration tests:**
`pytest -m integration` (needs a running VLA server).

**Linux + GPU notes**

- ManiSkill wants a working **Vulkan** stack for typical setups (see ManiSkill docs).
- `MANISKILL_SIM_BACKEND` defaults to `cpu` on Windows and unset (GPU) on Linux; override in `.env` only if needed.
- `**VLA_AXIS_REMAP` / `VLA_AXIS_SIGNS**` in `.env.example` were tuned for **Windows CPU** PickCube; re-tune on Linux/GPU if motion looks wrong.

---

More detail: `SRD.md`.
