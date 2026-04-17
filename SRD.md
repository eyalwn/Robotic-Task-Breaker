# Longor SRD

Private codebase: dependencies in `requirements.txt`. Source packages live under `src/`. Run with `PYTHONPATH=src`; tests use `pytest.ini` `pythonpath = src`.

## Goal

Enable **long-horizon** robot tasks by running an **LLM in a closed loop**:

1. The LLM **plans** and maintains an ordered list of **subtasks** (Python `list`).
2. For each subtask it selects and invokes **tools** only through the LLM (no separate programmatic executor). Arm manipulation uses **VLAs**: behavior is exposed via **parameterized tool factories** (e.g. a factory that builds a LangChain tool bound to a policy id, timeouts, and prompts)—not arbitrary codegen or `exec` of user Python.
3. **Feedback** is **RGB only** (sim renderer or real camera): capture → LLM / VLA as needed, then continue until the top-level task is done.
4. **Subtask success or failure** is judged by the **LLM** when a subtask ends. Finer replanning vs retry policy is **out of scope** for this document.

**Platforms (non-exhaustive):** the codebase stays **platform-generic**; near-term targets are **ManiSkill** and **LIBERO** simulation. The **Linux** reference setup assumes **NVIDIA + CUDA** (see `README.md`). **Windows (Git Bash)** supports the **app** venv and, for Octo **`octo-small-1.5`**, a dedicated Windows inference requirements file that **avoids** upstream `tensorflow_text` (no PyPI wheels for common Windows/Python combos) — see `README.md` §3 and `services/requirements-vla-octo-small-1.5-windows.txt`.

**Arm control:** manipulation policies are **VLAs only**, behind a **single VLA HTTP server** at a time (see below). Other stacks (navigation, grippers without VLA) are ordinary tools/backends as the codebase evolves.

## Milestones (first)

### M1 (implemented — Linux validation pending, Michael)

**Linux** support for the existing sim + VLA + agent stack (**parity** with Windows for dev/CI).

**Scope**

- Documented setup paths for **Linux (with CUDA)** (POSIX venv, ManiSkill, JAX+VLA on **NVIDIA**) without breaking the **Windows (Git Bash)** flow on the reference machine.
- A fresh clone + `README.md` steps should get a developer to: **two venvs**, **Octo HTTP service** running, **`examples/maniskill_pickcube.py`** against ManiSkill + VLA, and **`pytest`** (same expectations as Windows; **integration** tests are marked — see `pytest.ini` / `pytest -m integration`).

**Acceptance (manual, on Linux host)**

- [ ] `.venv`: `pip install -r requirements.txt` (+ sim deps), ManiSkill for PickCube per `README.md` / [ManiSkill installation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html), run `pytest` from repo root.
- [ ] `.vla-venv`: Octo per `README.md` §3 — **Linux:** Octo `requirements.txt` + `services/requirements-vla-octo-small-1.5-linux.txt`; **Windows:** `services/requirements-vla-octo-small-1.5-windows.txt` + `pip install -e ~/octo --no-deps` (no upstream `requirements.txt`); **`VLA_POLICY`** set; weights from Hugging Face in `services/octo_service.py`; `python services/run_vla_server.py` serves `/health` on `VLA_PORT`.
- [ ] With both terminals up and `.env` filled, `python examples/maniskill_pickcube.py` completes a run (success/failure per sim/policy is fine; wiring must work).
- [ ] `README.md` remains valid for **Windows (Git Bash)** and **Linux (with CUDA)** including Octo VLA venv paths in §3.

**Out of scope for M1**

- CI matrix on Linux (optional follow-up).
- New backends or changing the VLA HTTP contract.

### M2 (implemented — Linux validation pending, Michael)

**Additional VLA backend** — **π₀.5** (Pi0.5) via Physical Intelligence's [OpenPI](https://github.com/Physical-Intelligence/openpi) — proving the VLA HTTP contract is model-agnostic.

**Scope**

- `services/pi05_service.py`: FastAPI service wrapping OpenPI Pi0.5 inference, same `/health` + `POST /predict` contract as `octo_service.py`. Server-side maps Pi0.5's native action output to the `[dx, dy, dz, gripper]` wire format.
- `services/run_vla_server.py` supports `VLA_POLICY=pi05`.
- `services/requirements-vla-pi05-linux.txt` for the Pi0.5 `.vla-venv`.
- Setup instructions in `README.md` for **Linux + NVIDIA GPU** (≥ 8 GB VRAM). Pi0.5 does **not** run on Windows.
- `src/vla/` client code contains no model-specific assumptions (remove any Octo-specific references).

**Acceptance (manual, on Linux + NVIDIA host)**

- [ ] `.vla-venv` with OpenPI + service deps installed per `README.md`; `python services/run_vla_server.py` with `VLA_POLICY=pi05` starts and `GET /health` returns OK.
- [ ] `POST /predict` with a JPEG + task text returns a valid `{"actions": [[dx,dy,dz,gripper], ...]}` response.
- [ ] On **Windows**: `pytest` passes (unit tests mock the VLA service; no real Pi0.5 required).
- [ ] No Octo-specific references remain in `src/vla/`.

**Out of scope for M2**

- Pi0.5 task success on ManiSkill (wiring correctness only; the model is not trained on sim data).
- Pi0.5 on Windows (OpenPI requires Linux + NVIDIA GPU).
- Fine-tuning or model adaptation.

### M3

**LIBERO** simulation backend (same agent/VLA contract; env-specific adapters).

### M4

**Multi-tool long-horizon loop** on **simulation** (**ManiSkill** first, **Windows** reference host): LLM + ordered **subtasks** + **more than one** distinct **tool** used across the run (not a single fixed manipulation tool only), **RGB** feedback, **VLA** manipulation via parameterized factories and **one** VLA HTTP server at a time.

**M1–M3** widen OS, sim, and policy options; **M4** is the integrated loop milestone on the reference stack.

## VLA integration — architecture

The **client** (`vla` + sim/robot backends) talks to a **single VLA service process** over HTTP. The service implementation may swap (small vs large model); the **wire contract** stays generic.

```
┌─────────────────────┐   HTTP    ┌───────────────────────────┐
│  Agent / example    │◄─────────►│  VLA server (1 process)   │
│  (app venv)         │           │  loads policy weights     │
│  vla.* + backend    │           │  POST predict, GET health │
│  RGB in, actions    │           │                           │
│  applied via arm API│           └───────────────────────────┘
└─────────────────────┘
```

**Intent (not tied to one model):**

- **Health:** e.g. `GET /health` → JSON indicating readiness (and optionally model id).
- **Predict:** e.g. `POST` with **JPEG** image + optional **task** text → **action** for the arm (format and dimensionality are defined per deployment; the `vla` layer maps into the backend’s `move_ee` / gripper contract).

Implementation details (ports, batching, chunking, inner-step limits) live in code and `.env`, not in this SRD.

## Architecture constraints

- **`core/**`** stays **OS- and backend-agnostic** (no device paths, no sim imports, no robot SDKs).
- **Simulation and hardware** live in **`sim/**`, `robots/**`,** or dedicated backend modules; select backend via **config/env**.
- **Optional** heavy or HW-only dependencies: **lazy imports** and/or `requirements-optional.txt` so minimal sim dev does not install everything.
- Adding a new sim or robot: **adapters + tool factories**; avoid changing the LLM loop unless necessary.

## Configuration

Use **`.env`** at the repository root (see `.env.example`). Typical families of variables:

- **Backend / env:** which sim or robot, task name, display/render flags.
- **LLM:** model id, API keys (never committed).
- **VLA:** `VLA_POLICY` (which server implementation), `VLA_PORT`, timeouts, and calibration env vars consumed by `vla` (see `vla/config.py`).

Exact names evolve in code; the SRD only requires **config-driven** selection, not hard-coded paths.

## Testing

- **Unit tests** hermetic where possible (no live sim, no network); use fakes/mocks for backends and VLA HTTP.
- Shared fixtures: `tests/conftest.py`.
- **`pytest.ini`** sets `pythonpath = src`.
