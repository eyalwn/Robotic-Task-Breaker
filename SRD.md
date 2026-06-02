# Longor SRD

Private codebase: dependencies in `requirements.txt`. Source packages live under `src/`. Run with `PYTHONPATH=src`; tests use `pytest.ini` `pythonpath = src`.

## Goal

Enable robots to perform **long-horizon** tasks by separating **planning** (LLM) from **execution** (interpreter), so the agent reasons over the *structure* of the work, not just an ordered step list.

1. The **planner** (LLM) emits a **plan**: a structured algorithm with control flow (sequencing, conditionals, loops, feedback points) — not a flat list of subtasks.
2. The **executor** walks the plan and invokes **skills** — parameterized callable capabilities the planner composes. Skills include perception / state queries (returning structured data) and arm manipulation (wrapping VLAs).
3. **Feedback points** in the plan capture an RGB snapshot, let the LLM evaluate progress against the goal, and route the run to continue, stop, or replan. Replanning is bounded.
4. Plans are **declarative data, not executable Python**: arbitrary code execution is out of scope on purpose.

**Platforms.** The codebase stays **platform-generic**. The **primary E2E target** is **LIBERO** on **Linux + NVIDIA + CUDA**, where serious VLA evaluation happens. **ManiSkill on Windows (Git Bash)** is a **cross-platform smoke test** only: it verifies the planner / executor / skill code path stays alive on the dev host; it is not a target for VLA task success. Recursive sub-planning (a "composite" skill that re-invokes the planner) is a possible future extension, **not in current scope**.

**Arm control:** manipulation skills are **VLAs only**, behind a **single VLA HTTP server** at a time (see below). Other capabilities (navigation, simple grippers) are ordinary skills as the codebase evolves.

## Milestones (first)

### M1 (implemented — Linux validation pending)

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

### M2 (implemented — Linux validation pending)

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

### M3 (implemented — Linux validation pending)

**LIBERO** simulation backend via [LeRobot](https://huggingface.co/docs/lerobot/main/libero) — same agent/VLA contract, env-specific adapters. Initial target: **LIBERO-Object** suite.

**Scope**

- `src/sim/libero_backend.py`: `create_env()`, `LiberoCamera`, `LiberoArm` adapters matching the `ManiSkillCamera`/`ManiSkillArm` interface (`capture_image`, `move_ee`, `set_gripper`). LIBERO uses MuJoCo + OSC controllers — direct EE delta control (no Jacobian IK needed).
- `src/sim/libero_success_checker.py`: reads LIBERO's success flag from env info.
- `examples/libero_object.py`: example script wiring LIBERO-Object + VLA, same pattern as `maniskill_pickcube.py`.
- `requirements-libero.txt`: LeRobot + LIBERO deps (**Linux only**; LIBERO requires `sys_platform == 'linux'`).
- `arm_tools.py` must work unchanged with `LiberoArm`.

**Acceptance (manual, on Linux host)**

- [ ] `.venv` with LeRobot + LIBERO deps from `requirements-libero.txt`; `export MUJOCO_GL=egl`.
- [ ] `python examples/libero_object.py` with a running VLA server starts a LIBERO-Object task, drives the arm via VLA, and runs to completion (wiring correctness; task success not required).
- [ ] On **Windows**: `pytest` passes (LIBERO backend unit tests mock the env; no real LIBERO required).
- [ ] `arm_tools.py` works unchanged with `LiberoArm`.

**Out of scope for M3**

- LIBERO on Windows (Linux only).
- LIBERO-Long or other suites beyond LIBERO-Object.
- Centralized backend dispatcher (examples import their backend directly).

### M4 (implemented)

**Plan AST + executor + skill registry** — pure software layer; no sim, no real LLM.

**Scope**

- `src/core/skills.py`: `Skill` base, `SkillRegistry`, shared typed values (e.g. `Pose`, `DetectedObject`). Interfaces only; concrete implementations live in backend packages.
- `src/core/plan/`: `schema.py` (AST schema), `interpreter.py` (walker + restricted expression evaluator over a small typed value space; **no `eval` / `exec`**), `planner.py` (LLM-facing interface; concrete LLM call deferred to M6), `feedback.py`, `executor.py`.
- Signals: `ReturnSignal(value)` from `return` nodes → `status="done"`; `StopSignal()` from `feedback → STOP` → `status="stopped"`.
- Skill adapters: `arm.grab` / `arm.place` in `src/vla/arm_skills.py`; `vla_tool` injected as a callable accepting `task` kwarg.
- Cross-platform; no sim or VLA contract changes.

**Acceptance**

- [x] Schema validation accepts well-formed plans and rejects malformed ones with usable errors.
- [x] Interpreter handles every node type (`seq`, `if`, `while`, `call`, `feedback`, `return`) on hand-crafted plans.
- [x] Restricted expression evaluator rejects unsafe input (no attribute access on arbitrary objects, no name lookup outside bound vars, no calls).
- [x] Handcrafted-plan E2E test runs against fully mocked sim + mocked LLM evaluator, exercising both `continue` and `replan` feedback decisions.
- [x] No changes to the VLA HTTP contract.

**Out of scope for M4**

- Real perception, real LLM-emitted plans, real sim integration, recursive sub-planning.

### M5 (implemented — Linux validation pending)

**Sim-GT skills + handcrafted E2E** on **LIBERO** (Linux primary) and **ManiSkill** (Windows smoke test).

**Scope**

- `src/sim/libero_skills.py`: `perception.detect` / `perception.locate` / `state.list_objects` against LIBERO's MuJoCo scene. `_unwrap_to_sim` walks the LeRobot SyncVectorEnv wrapper chain; skills return `None`/`[]` gracefully if sim is unreachable.
- `src/sim/maniskill_skills.py`: same three skills against ManiSkill scene via `_find_actors` reflection on `env.unwrapped`.
- Sim selection by example import; no config knob.
- `examples/long_horizon_handcrafted_libero.py`: handcrafted plan + `_AlwaysContinue` feedback stub + `_NoReplan` planner, wired to `Executor`.
- `examples/long_horizon_handcrafted_maniskill.py`: same, ManiSkill PickCube — Windows smoke test.

**Acceptance**

- [ ] On **Linux**: handcrafted plan completes the chosen LIBERO task end-to-end against LIBERO + a running VLA server.
- [ ] On **Windows**: ManiSkill version starts and executes ≥1 skill call without crashing (task success not required; arm calls may fail if no VLA is running).
- [ ] `arm.grab` / `arm.place` operate over the unchanged VLA HTTP contract.

**Out of scope for M5**

- Real perception (and the `PerceptionBackend` abstraction it motivates), LLM-emitted plans, LIBERO-Long.

### M6 (implemented — Linux validation pending)

**LLM-emitted plans, full E2E on LIBERO-Long.**

**Scope**

- `src/core/plan/llm_planner.py`: `LLMPlanner(model, max_plan_retries=3)` — real LLM call via tool-calling with a single tool `emit_plan(plan_json: str)`; validates with `validate_plan`; retries on `ValidationError` with the error fed back, capped at `max_plan_retries`.
- `src/core/plan/llm_feedback.py`: `LLMFeedbackEvaluator(model)` — sends RGB snapshot + description to LLM via `emit_decision(decision, reason?)` tool-call; returns `FeedbackDecision`.
- `src/core/plan/trace.py`: `TraceWriter(trace_dir, goal)` — appends JSONL events (`plan_start`, `skill_call`, `skill_result`, `feedback`, `replan`, `plan_end`) to a per-run file named `{timestamp}_{goal_slug}.jsonl` in `trace_dir`.
- `src/core/plan/interpreter.py` *(minimal change)*: add optional `trace: TraceWriter | None` param; emit `skill_call`/`skill_result`/`feedback` events when set.
- `src/core/plan/executor.py` *(minimal change)*: add optional `trace_dir: str | None` param; create/pass `TraceWriter` per run; emit `plan_start`/`replan`/`plan_end` events.
- Skill catalog in LLM prompt auto-rendered from `SkillRegistry.render_catalog()` — single source of truth.
- `examples/long_horizon_llm_libero.py`: full E2E on **LIBERO-Long / LIBERO-10** with `LLMPlanner` + `LLMFeedbackEvaluator` + LIBERO sim-GT skills + arm skills + `Executor`. Target task: LIBERO-10 task 0 (implementing agent confirms exact task name at scoping). Trace written to `traces/` by default.

**Acceptance**

- [ ] On **Linux + LIBERO-Long**: LLM emits a valid plan from the task language goal, the executor runs it with the VLA in the loop, and the run reaches at least one happy-path completion of the chosen task.
- [x] Schema-violating LLM output triggers a bounded retry with the validation error injected into the next prompt.
- [x] Replan and step-budget caps are enforced.
- [x] On **Windows**: `pytest` passes (160 passed, 5 skipped).
- [x] Per-run trace file written to `traces/` and contains all expected event types.

**Out of scope for M6**

- Real perception, success across all LIBERO-10 tasks (one task is enough), recursive sub-planning.

### Post-Linux-validation cleanup (planned)

Retire the legacy flat-loop agent path once Linux validation of M1–M6 is confirmed.

**Scope**

- Delete `src/core/llm_agent.py` and `src/sim/runner.py`.
- Update `examples/libero_object.py` and `examples/maniskill_pickcube.py` to use the M5/M6 executor path (or remove them if the M5 handcrafted examples fully cover their acceptance criteria).
- Delete/rewrite any tests that exclusively exercise `LLMAgent` or `run_sim_agent` (e.g. `tests/test_e2e_sim_loop.py`, `tests/test_sim_runner.py`).
- Update M1–M3 acceptance criteria to reference the new examples.

**Precondition:** Linux validation of M1–M6 complete.

---

### M7 (optional, later)

**Real perception backend.** Open-vocab detector (e.g. Grounding DINO or OWL-ViT + SAM2) implementing the same `perception.*` skill set as the sim-GT versions, but against rendered RGB instead of sim ground truth — so a single sim can be paired with either sim-GT or real perception. Introduces a `PerceptionBackend` interface in `src/core/skills.py` and a config knob (e.g. `PERCEPTION_BACKEND=sim_gt | openvocab`). Linux + GPU. Out of current planning scope; recorded so the option is on file.

**M1–M3** widen OS, sim, and policy options; **M4–M6** add the planner architecture on top, with **LIBERO** as the primary E2E target and **ManiSkill on Windows** as the cross-platform smoke test.

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
- **Plan representation:** structured JSON AST (schema in `src/core/plan/schema.py`); the executor walks it with a restricted expression evaluator. **No `exec`** of generated code; **no `eval`** of arbitrary expressions.
- **Skills:** `Skill` base + `SkillRegistry` live in `src/core/skills.py` (interfaces only). Concrete skill implementations live in their backend packages (`src/sim/*_skills.py`, `src/vla/arm_skills.py`, future `src/robots/*_skills.py`) and self-register at import time.
- Adding a new sim, robot, or capability: add a backend adapter and skill implementations in the relevant backend package; the planner / executor stay unchanged.
- **Legacy `core/llm_agent.py`** — the pre-planner monolithic agent loop. Stays in place during M4–M6 as the older code path; its retirement or rewrite on top of the new planner is out of current scope.

## Configuration

Use **`.env`** at the repository root (see `.env.example`). Typical families of variables:

- **Backend / env:** which sim or robot, task name, display/render flags.
- **LLM:** model id, API keys (never committed).
- **VLA:** `VLA_POLICY` (which server implementation), `VLA_PORT`, timeouts, and calibration env vars consumed by `vla` (see `vla/config.py`).
- **Planner:** replan and step-budget caps (defaults ≤3 replans per run, ~30 skill calls per plan).

Exact names evolve in code; the SRD only requires **config-driven** selection, not hard-coded paths.

## Testing

- **Unit tests** hermetic where possible (no live sim, no network); use fakes/mocks for backends and VLA HTTP.
- Shared fixtures: `tests/conftest.py`.
- **`pytest.ini`** sets `pythonpath = src`.
