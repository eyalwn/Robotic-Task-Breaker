# Longor SRD

Private codebase: dependencies in `deps/requirements.txt`; install the repo editable with `pip install -e .` so `core`, `sim`, `vla`, and `robots` import without `PYTHONPATH`. Source packages live under `src/`. Tests use `pyproject.toml` `[tool.pytest.ini_options]` `pythonpath = src`.

## Goal

Enable robots to perform **long-horizon** tasks by separating **planning** (LLM) from **execution** (interpreter), so the agent reasons over the *structure* of the work, not just an ordered step list.

1. The **planner** (LLM) emits a **plan**: a structured algorithm with control flow (sequencing, conditionals, loops, feedback points) — not a flat list of subtasks.
2. The **executor** walks the plan and invokes **skills** — parameterized callable capabilities the planner composes. Skills include perception / state queries (returning structured data) and arm manipulation (wrapping VLAs).
3. **Feedback points** in the plan capture an RGB snapshot, let the LLM evaluate progress against the goal, and route the run to continue, stop, or replan. Replanning is bounded.
4. Plans are **declarative data, not executable Python**: arbitrary code execution is out of scope on purpose.

**Platforms.** The codebase stays **platform-generic**. The **primary E2E target** is **LIBERO** on **Linux + NVIDIA + CUDA**, where serious VLA evaluation happens. **ManiSkill on Windows (Git Bash)** is a **cross-platform smoke test** only: it verifies the planner / executor / skill code path stays alive on the dev host; it is not a target for VLA task success. Recursive sub-planning (a "composite" skill that re-invokes the planner) is a possible future extension, **not in current scope**.

**Arm control:** manipulation skills are **VLAs only**, behind a **single VLA HTTP server** at a time (see below). Other capabilities (navigation, simple grippers) are ordinary skills as the codebase evolves.

## Milestones



### Completed foundations (historical, unchanged IDs)

- **M1 (done):** Linux parity for the existing sim + VLA + test stack.
- **M2 (done):** Pi0.5 backend (`VLA_POLICY=pi05`) validated on Linux GPU.
- **M3 (done):** LIBERO backend integration (`LiberoBackend`, Linux-only runtime).
- **M4 (done):** plan AST + executor + skill registry + safe expression evaluator.
- **M5 (done):** handcrafted plan E2E with sim-GT perception + arm skills.
- **M6 (done):** LLM planner + LLM feedback + trace writer + `Runner`/`Backend` architecture.
- **M8 (done):** SmolVLA backend (`VLA_POLICY=smolvla`) validated on Colab T4.



### Evaluation doctrine (applies to all future milestones)

- Do **not** conclude policy incapability from monolithic LIBERO-10 runs alone.
- Always prove wiring on simpler isolated conditions before long-horizon target tasks.
- For SmolVLA baseline checks, closed-loop `n_action_steps=1` is mandatory before trying larger horizons.
- Keep debug scope contained: experiments and sweeps scripts belong in `validation/**`, not in production runtime modules.



### Current execution order (replanned)



### M7 (planned) — Wiring correctness ladder (isolation first)

**Goal**

Prove Longor wiring is correct in progressively harder, isolated stages before revisiting long-horizon real-perception acceptance.

**Scope**

1. **M7.1 cleanup pass (done)**
  - Remove non-essential post-M8 debug drift from hot-path files (`src/vla/tools.py`, `src/sim/libero_backend.py`, `services/smolvla_service.py`) while preserving validated contract behavior.
  - Keep runtime logic thin; move experiment orchestration to `validation/**`.
2. **M7.2 isolated arm-control probe (sim-GT) (done)**
  - Simplest LIBERO environment, no planner/executor/LLM loop.
  - Entrypoint: `validation/m7_wiring_probe.py --gate m7.2` (default suite `libero_object`, task 0).
  - Ask VLA to move EE to a target pose/region; verify deterministic motion and termination behavior.
3. **M7.3 isolated arm-control probe (real-perception mode)**
  - Same probe as M7.2 with `PERCEPTION_BACKEND=real`.
  - Ensure no silent fallback to sim-GT registration.
4. **M7.4 direct task probe, no planner decomposition**
  - Simplest LIBERO pick/place task.
  - Pass only task text + required env observation to VLA tool path (no LLM planning loop).
5. **M7.5 structured-plan probe without LLM**
  - Inject a static JSON plan (`StaticPlanner`) and validate executor/skill orchestration on the same simple task.
6. **M7.6 full decomposition probe**
  - Re-enable LLM planner+feedback on the simple task and confirm decomposition succeeds before moving back to LIBERO-10.

**Acceptance**

- Every gate (M7.1–M7.6) has a dedicated script entrypoint that prints a concise pass/fail summary with key metrics to stdout and exits 0 on pass.
- `pytest` is green on Windows after M7.1 cleanup.
- M7.2 and M7.3 show consistent EE motion and no runtime crashes.
- M7.4–M7.6 produce at least one successful simple-task run each.

**Out of scope for M7**

- LIBERO-10 breadth (campaign).
- Hardware backend milestones.
- Changing VLA HTTP wire format.



### M9 (planned; former M7) — Real perception E2E on LIBERO-10 task 0

**Goal**

After M7 proves wiring, run the original M7 target: `examples/long_horizon_llm_libero.py` on `libero_10` task 0 with real perception and SmolVLA.

**Scope**

- Real-perception skill path (`PERCEPTION_BACKEND=real`) only.
- SmolVLA service path (`VLA_POLICY=smolvla`) only for acceptance.
- Preserve planner/executor/VLA contracts.

**Acceptance**

- Linux + Colab T4 run reaches `RunResult.success=True` on task 0.
- Runtime path used for success does not register sim-GT perception skills.
- Validation script prints a concise pass/fail summary with key run metrics to stdout and exits 0 on pass.



### M10 (planned; former M9) — Sim-GT perception removal (strict)

Delete runnable sim-GT perception registration and config fallbacks. Real perception becomes the only LIBERO runtime path.

### M11 (planned; former M10) — LIBERO-10 campaign

Run independent per-task evaluations (`task_idx` 0..9), with per-task pass/fail stdout summaries and at least one success per task.

### M12 (planned; former M11) — XLeRobot-like simulation backend

Add a simulation backend that mirrors XLeRobot capabilities while preserving the `Runner` + `Backend` contracts.

### M13 (planned; former M12) — Architecture hardening and cleanup

Deep architecture review + targeted redundancy deletion before physical hardware milestones.

### M14 (planned; former M13) — Real XLeRobot hardware backend

Implement production hardware backend with strict pre-flight and safety gates.

### M15 (planned; former M14) — Real-world XLeRobot task validation

At least one full autonomous success on the physical robot with archived evidence.

---



## Validation tooling (Colab)

Colab notebooks are **launchers only**.

- Notebook responsibilities:
  1. collect secrets/env,
  2. clone/sync repo and install deps,
  3. start/poll required services,
  4. run exactly one script under `validation/**`,
  5. print pass/fail summary from script stdout.
- Experiment orchestration, sweeps, and verdict logic belong in Python scripts, not notebook cells.
- Validation notebooks clone the repository default branch only (no `GITHUB_BRANCH` flow).

**Script output contract**

Every Colab-launched script must:

- print a concise pass/fail summary with key metrics to stdout,
- exit `0` on pass, non-zero on fail.

**Notebook-to-script mapping (planned target state)**


| Notebook                               | Script                                   |
| -------------------------------------- | ---------------------------------------- |
| `colab/m7_wiring_probe.ipynb`          | `validation/m7_wiring_probe.py`          |
| `colab/m9_real_perception_task0.ipynb` | `validation/m9_real_perception_task0.py` |
| `colab/m11_libero_10_campaign.ipynb`   | `validation/m11_libero_10_campaign.py`   |


Legacy notebooks remain temporary until migration to the launcher model is complete.

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

- `**core/**`** stays **OS- and backend-agnostic** (no device paths, no sim imports, no robot SDKs).
- **Simulation and hardware** live in `**sim/`**,** `robots/`****,** or dedicated backend modules; select backend via **config/env**. Each backend implements the `Backend` protocol (`src/core/backend.py`) and owns its own sim/VLA wiring so example scripts stay a thin application layer (`Runner().run(backend, goal)`).
- `**vla/**`** is sim-agnostic (no `sim` imports); backend-specific behavior is injected via hooks (e.g. `name_mapper` / `place_result_hook` on `register_arm_skills`).
- **Optional** heavy or HW-only dependencies: **lazy imports** and/or `deps/requirements-optional.txt` so minimal sim dev does not install everything.
- **Plan representation:** structured JSON AST (schema in `src/core/plan/schema.py`); the executor walks it with a restricted expression evaluator. **No** `exec` of generated code; **no** `eval` of arbitrary expressions.
- **Skills:** `Skill` base + `SkillRegistry` live in `src/core/skills.py` (interfaces only). Concrete skill implementations live in their backend packages (`src/sim/*_skills.py`, `src/vla/arm_skills.py`, future `src/robots/*_skills.py`) and self-register at import time.
- Adding a new sim, robot, or capability: add a `Backend` implementation + skill implementations in the relevant backend package; `Runner`, the planner, and the `Executor` stay unchanged.
- `core/llm_agent.py` — the general-purpose LLM tool-call loop used by HW robot agents (`XLeRobot`). Not a sim concern; retained as-is.



## Configuration

Use `**.env**` at the repository root (see `.env.example`). Typical families of variables:

- **Backend / env:** which sim or robot, task name, display/render flags.
- **LLM:** model id, API keys (never committed).
- **VLA:** `VLA_POLICY` (which server implementation), `VLA_PORT`, timeouts, and calibration env vars consumed by `vla` (see `vla/config.py`).
- **Planner:** replan and step-budget caps (defaults ≤3 replans per run, ~30 skill calls per plan).

Exact names evolve in code; the SRD only requires **config-driven** selection, not hard-coded paths.

## Testing

- **Unit tests** hermetic where possible (no live sim, no network); use fakes/mocks for backends and VLA HTTP.
- Shared fixtures: `tests/conftest.py`.
- `**pyproject.toml`** `[tool.pytest.ini_options]` sets `pythonpath = src`.

