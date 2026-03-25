# Longor SRD

Private codebase: dependencies in `requirements.txt`. Source packages live under `src/`. Run with `PYTHONPATH=src`; tests use `pytest.ini` `pythonpath = src`.

## Goal

Enable **long-horizon** robot tasks by running an **LLM in a closed loop**:

1. The LLM **plans** and maintains an ordered list of **subtasks** (Python `list`).
2. For each subtask it selects and invokes **tools** only through the LLM (no separate programmatic executor). Arm manipulation uses **VLAs**: behavior is exposed via **parameterized tool factories** (e.g. a factory that builds a LangChain tool bound to a policy id, timeouts, and prompts)—not arbitrary codegen or `exec` of user Python.
3. **Feedback** is **RGB only** (sim renderer or real camera): capture → LLM / VLA as needed, then continue until the top-level task is done.
4. **Subtask success or failure** is judged by the **LLM** when a subtask ends. Finer replanning vs retry policy is **out of scope** for this document.

**Platforms (non-exhaustive):** the codebase stays **platform-generic**; planned targets are **ManiSkill** simulation (including XLeRobot-style assets where applicable), **LIBERO** simulation, and **real XLeRobot** hardware.

**Arm control:** manipulation policies are **VLAs only**, behind a **single VLA HTTP server** at a time (see below). Other stacks (navigation, grippers without VLA) are ordinary tools/backends as the codebase evolves.

## Milestones (first)

### M1

End-to-end **long-horizon loop** on **simulation only**, **Windows**, **ManiSkill** first: LLM + subtask list + parameterized manipulation tools + one VLA server + RGB.

### M2

**Linux** support for the same sim + VLA stack (parity with Windows for dev/CI).

### M3

**LIBERO** simulation backend (same agent/VLA contract; env-specific adapters).

### M4

**Additional VLA backends** (including heavier models than the initial lightweight server—stronger hosts). Same HTTP-facing contract where possible.

### M5

**Real XLeRobot** hardware path (RGB, VLA tools, backends)—subject to safety and bring-up work not specified here.

Order and numbering may be adjusted; **M2–M4** are urgent parallel concerns after **M1** is demonstrable.

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
