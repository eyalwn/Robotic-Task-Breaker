# Longor SRD

Private codebase: dependencies in `requirements.txt`. Source packages live under `src/`. Run with `PYTHONPATH=src`; tests use `pytest.ini` `pythonpath = src`.

## Goal

Enable **long-horizon** robot tasks by running an **LLM in a closed loop**:

1. The LLM **plans** and maintains an ordered list of **subtasks** (Python `list`).
2. For each subtask it selects and invokes **tools** only through the LLM (no separate programmatic executor). Arm manipulation uses **VLAs**: behavior is exposed via **parameterized tool factories** (e.g. a factory that builds a LangChain tool bound to a policy id, timeouts, and prompts)—not arbitrary codegen or `exec` of user Python.
3. **Feedback** is **RGB only** (sim renderer or real camera): capture → LLM / VLA as needed, then continue until the top-level task is done.
4. **Subtask success or failure** is judged by the **LLM** when a subtask ends. Finer replanning vs retry policy is **out of scope** for this document.

**Platforms (non-exhaustive):** the codebase stays **platform-generic**; near-term targets are **ManiSkill** and **LIBERO** simulation.

**Arm control:** manipulation policies are **VLAs only**, behind a **single VLA HTTP server** at a time (see below). Other stacks (navigation, grippers without VLA) are ordinary tools/backends as the codebase evolves.

## Milestones (first)

### M1

**Linux** support for the existing sim + VLA + agent stack (**parity** with Windows for dev/CI).

### M2

**Additional VLA backends** beyond the first HTTP service (e.g. **π₀.5** / Pi0.5-class models on capable hosts). Prefer a **stable HTTP-facing contract** where practical.

### M3

**LIBERO** simulation backend (same agent/VLA contract; env-specific adapters).

### M4

**Multi-tool long-horizon loop** on **simulation** (**ManiSkill** first, **Windows** reference host): LLM + ordered **subtasks** + **more than one** distinct **tool** used across the run (not a single fixed manipulation tool only), **RGB** feedback, **VLA** manipulation via parameterized factories and **one** VLA HTTP server at a time.

**M1–M3** widen OS, sim, and policy options; **M4** is the integrated loop milestone on the reference stack.

## VLA integration — architecture

The **client** (`vla` + sim/robot backends) talks to a **single VLA service process** over HTTP. The service implementation may swap (small vs large model); the **wire contract** stays generic.

```
┌─────────────────────┐   HTTP    ┌──────────────────────────┐
│  Agent / example    │◄─────────►│  VLA server (1 process)   │
│  (app venv)         │           │  loads policy weights     │
│  vla.* + backend    │           │  POST predict, GET health │
│  RGB in, actions    │           │                           │
│  applied via arm API│           └──────────────────────────┘
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
- **VLA:** base URL for the HTTP service, timeouts, and any calibration env vars consumed by `vla` (see `vla/config.py`).

Exact names evolve in code; the SRD only requires **config-driven** selection, not hard-coded paths.

## Testing

- **Unit tests** hermetic where possible (no live sim, no network); use fakes/mocks for backends and VLA HTTP.
- Shared fixtures: `tests/conftest.py`.
- **`pytest.ini`** sets `pythonpath = src`.
