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



### Completed foundations (historical)

- **M1 (done):** Linux parity for the existing sim + VLA + test stack.
- **M2 (done):** Pi0.5 backend (`VLA_POLICY=pi05`) validated on Linux GPU.
- **M3 (done):** LIBERO backend integration (`LiberoBackend`, Linux-only runtime).
- **M4 (done):** plan AST + executor + skill registry + safe expression evaluator.
- **M5 (done):** handcrafted plan E2E with sim-GT perception + arm skills.
- **M6 (done):** LLM planner + LLM feedback + trace writer + `Runner`/`Backend` architecture.
- **SmolVLA backend baseline (done; former M8):** `VLA_POLICY=smolvla` validated on Colab T4.



### Evaluation doctrine (applies to all future milestones)

- Do **not** conclude policy incapability from monolithic LIBERO-10 runs alone.
- Always prove wiring on simpler isolated conditions before long-horizon target tasks.
- For SmolVLA baseline checks, closed-loop `n_action_steps=1` is mandatory before trying larger horizons.
- Keep debug scope contained: experiments and sweeps scripts belong in `validation/**`, not in production runtime modules.
- From **M8 onward**, validation entrypoints must have **zero dependency** on `src/sim/libero_smoke.py` (no imports, no helper reuse).



### Current execution order (replanned)



### M7 (done) — Isolated arm-control wiring probe (sim-GT)

**Goal**

Prove the VLA->arm wiring path is alive in isolation.

**Scope**

- Simplest LIBERO environment, no planner/executor/LLM loop.
- Entrypoint: `validation/m7_wiring_probe.py --gate m7.2` (default suite `libero_object`, task 0).
- Validate wiring only: VLA request/response, action application, and bounded run termination.
- Explicitly out of scope for M7: policy quality claims (for example, expected EE->target distance-decay magnitude).

**Acceptance**

- Script prints a concise pass/fail summary with key metrics and exits `0` on pass.
- Probe shows non-zero EE motion above threshold and no runtime crash.

**Historical note**

- Former **M7.3** (real-perception wiring probe) is complete and remains a wiring gate only; it is not a policy-health proof.



### M8 (planned) — Isolated SmolVLA health proof (same conditions as M7.2)

**Goal**

Prove policy health under the same isolated setup as M7.2: SmolVLA should approach decisively, close, lift, and finish successfully within a reasonable step budget.

**Scope**

- Use the same isolation constraints as M7.2 (no planner/executor/LLM).
- Add policy-health diagnostics beyond wiring: per-step EE->target distance, action magnitudes, gripper state transitions, and success reason.
- Entrypoint: `validation/m8_policy_health_probe.py`.
- Zero dependency on `src/sim/libero_smoke.py`.

**Acceptance**

- At least one seeded run reaches a successful lift/task-complete outcome within the configured step budget.
- Output includes a clear per-step distance trace suitable for auditing expected approach behavior.
- Script prints concise pass/fail metrics and exits `0` on pass.



### M9 (planned; former M7.4) — Direct task probe, no planner decomposition

**Goal**

Validate direct VLA task success on the simplest LIBERO pick task with no planning layer.

**Scope**

- Pass only task text + required env observation to the VLA tool path.
- No planner/executor/LLM loop.
- Zero dependency on `src/sim/libero_smoke.py`.

**Acceptance**

- At least one successful simple-task run (`reason=task_success`, `status=success`).
- Script prints concise pass/fail metrics and exits `0` on pass.



### M10 (planned; former M7.5) — Structured-plan probe without LLM

**Goal**

Validate executor/skill orchestration on the same simple task using a static plan only.

**Scope**

- Inject a static JSON plan (`StaticPlanner`) for the simple task.
- Keep planner LLM disabled; test executor + skill wiring only.
- Zero dependency on `src/sim/libero_smoke.py`.

**Acceptance**

- At least one successful simple-task run through static-plan execution.
- Script prints concise pass/fail metrics and exits `0` on pass.



### M11 (planned; former M7.6) — Full decomposition probe

**Goal**

Re-enable planner+feedback decomposition on the same simple task and verify success before returning to LIBERO-10.

**Scope**

- Re-enable LLM planner + LLM feedback evaluator.
- Keep task fixed to the simple probe target; no LIBERO-10 breadth yet.
- Zero dependency on `src/sim/libero_smoke.py`.

**Acceptance**

- At least one successful decomposed run on the simple task.
- Script prints concise pass/fail metrics and exits `0` on pass.

**Out of scope for M8-M11**

- LIBERO-10 breadth (campaign).
- Hardware backend milestones.
- Changing VLA HTTP wire format.



### M12 (planned; former M9) — Real perception E2E on LIBERO-10 task 0

**Goal**

After M8-M11 prove policy health + decomposition on the simple task, run the original long-horizon target: `examples/long_horizon_llm_libero.py` on `libero_10` task 0 with real perception and SmolVLA.

**Scope**

- Real-perception skill path (`PERCEPTION_BACKEND=real`) only.
- SmolVLA service path (`VLA_POLICY=smolvla`) only for acceptance.
- Preserve planner/executor/VLA contracts.

**Acceptance**

- Linux + Colab T4 run reaches `RunResult.success=True` on task 0.
- Runtime path used for success does not register sim-GT perception skills.
- Validation script prints a concise pass/fail summary with key run metrics to stdout and exits 0 on pass.



### M13 (planned; former M10) — Sim-GT perception removal (strict)

Delete runnable sim-GT perception registration and config fallbacks. Real perception becomes the only LIBERO runtime path.

### M14 (planned; former M11) — LIBERO-10 campaign

Run independent per-task evaluations (`task_idx` 0..9), with per-task pass/fail stdout summaries and at least one success per task.

### M15 (planned; former M12) — XLeRobot-like simulation backend

Add a simulation backend that mirrors XLeRobot capabilities while preserving the `Runner` + `Backend` contracts.

### M16 (planned; former M13) — Architecture hardening and cleanup

Deep architecture review + targeted redundancy deletion before physical hardware milestones.

### M17 (planned; former M14) — Real XLeRobot hardware backend

Implement production hardware backend with strict pre-flight and safety gates.

### M18 (planned; former M15) — Real-world XLeRobot task validation

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


| Notebook                          | Script                                                    |
| --------------------------------- | --------------------------------------------------------- |
| `colab/m7_wiring_probe.ipynb`     | `validation/m7_wiring_probe.py` / `validation/m8_policy_health_probe.py` / `validation/m7_direct_task_probe.py` |
| `colab/m12_real_perception_task0.ipynb` | `TBD` (planned: `validation/m12_real_perception_task0.py`) |
| `colab/m14_libero_10_campaign.ipynb`    | `TBD` (planned: `validation/m14_libero_10_campaign.py`)   |


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

