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

### M1 (implemented ✓)

**Linux** support for the existing sim + VLA + agent stack (**parity** with Windows for dev/CI).

**Scope**

- Documented setup paths for **Linux (with CUDA)** (POSIX venv, ManiSkill, JAX+VLA on **NVIDIA**) without breaking the **Windows (Git Bash)** flow on the reference machine.
- A fresh clone + `README.md` steps should get a developer to: **two venvs**, **Octo HTTP service** running, `**examples/long_horizon_handcrafted_maniskill.py`** against ManiSkill + VLA, and `**pytest**` (same expectations as Windows; **integration** tests are marked — see `pytest.ini` / `pytest -m integration`).

**Acceptance (manual, on Linux host)**

- `.venv`: `pip install -r requirements.txt` (+ sim deps), ManiSkill for PickCube per `README.md` / [ManiSkill installation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html), run `pytest` from repo root.
- `.vla-venv`: Octo per `README.md` §3 — **Linux:** Octo `requirements.txt` + `services/requirements-vla-octo-small-1.5-linux.txt`; **Windows:** `services/requirements-vla-octo-small-1.5-windows.txt` + `pip install -e ~/octo --no-deps` (no upstream `requirements.txt`); `**VLA_POLICY`** set; weights from Hugging Face in `services/octo_service.py`; `python services/run_vla_server.py` serves `/health` on `VLA_PORT`.
- With both terminals up and `.env` filled, `python examples/long_horizon_handcrafted_maniskill.py` completes a run (success/failure per sim/policy is fine; wiring must work).
- `README.md` remains valid for **Windows (Git Bash)** and **Linux (with CUDA)** including Octo VLA venv paths in §3.

**Colab validation results (Python 3.12, GPU — June 2026)**

- [x] `pytest` (non-integration, Linux): **160 passed, 2 skipped, 3 deselected** — Linux parity with Windows confirmed.
- [~] Octo VLA E2E on ManiSkill: **closed — N/A**. Octo's pinned deps are incompatible with Python 3.12 (`numpy==1.24.3` / `ml_dtypes==0.2.0` have no Py3.12 wheels; `jax.random.KeyArray` removed in JAX 0.4.24). The VLA HTTP contract (`/health` + `POST /predict`) was validated end-to-end by M2 (Pi0.5 on A100) and M3 (LIBERO wiring); M1's Octo E2E acceptance criterion is therefore superseded and will not be re-attempted. Working Colab recipe preserved in `colab/m1_maniskill_octo.ipynb` for reference.

**Out of scope for M1**

- CI matrix on Linux (optional follow-up).
- New backends or changing the VLA HTTP contract.

### M2 (implemented — Linux validated ✓)

**Additional VLA backend** — **π₀.5** (Pi0.5) via Physical Intelligence's [OpenPI](https://github.com/Physical-Intelligence/openpi) — proving the VLA HTTP contract is model-agnostic.

**Scope**

- `services/pi05_service.py`: FastAPI service wrapping OpenPI Pi0.5 inference, same `/health` + `POST /predict` contract as `octo_service.py`. Server-side maps Pi0.5's native action output to the `[dx, dy, dz, gripper]` wire format.
- `services/run_vla_server.py` supports `VLA_POLICY=pi05`.
- `services/requirements-vla-pi05-linux.txt` for the Pi0.5 `.vla-venv`.
- Setup instructions in `README.md` for **Linux + NVIDIA GPU** (≥ 8 GB VRAM). Pi0.5 does **not** run on Windows.
- `src/vla/` client code contains no model-specific assumptions (remove any Octo-specific references).

**Acceptance (manual, on Linux + NVIDIA host)**

- `.vla-venv` with OpenPI + service deps installed per `README.md`; `python services/run_vla_server.py` with `VLA_POLICY=pi05` starts and `GET /health` returns OK.
- `POST /predict` with a JPEG + task text returns a valid `{"actions": [[dx,dy,dz,gripper], ...]}` response.
- On **Windows**: `pytest` passes (unit tests mock the VLA service; no real Pi0.5 required).
- No Octo-specific references remain in `src/vla/`.

**Colab validation results (Python 3.12, A100 GPU — June 2026)**

- [x] Pi0.5 VLA service `/health` + `POST /predict`: **passed** — 15 action steps, `[dx,dy,dz,gripper]` format confirmed.
- [x] `pytest` (non-integration, Linux): **160 passed, 2 skipped, 3 deselected**.

Notes: requires A100 (free T4 OOM-kills during 11.6 GiB checkpoint load). Dep patches baked into `colab/m2_pi05_vla.ipynb`: pinned lerobot to OpenPI's git rev (`0cf8648`), upgraded `transformers`, set `timeout=180` for JIT warmup on first `/predict`.

**Out of scope for M2**

- Pi0.5 task success on ManiSkill (wiring correctness only; the model is not trained on sim data).
- Pi0.5 on Windows (OpenPI requires Linux + NVIDIA GPU).
- Fine-tuning or model adaptation.

### M3 (implemented — Linux validated ✓)

**LIBERO** simulation backend via [LeRobot](https://huggingface.co/docs/lerobot/main/libero) — same agent/VLA contract, env-specific adapters. Initial target: **LIBERO-Object** suite.

**Scope**

- `src/sim/libero_backend.py`: `create_env()`, `LiberoCamera`, `LiberoArm` adapters matching the `ManiSkillCamera`/`ManiSkillArm` interface (`capture_image`, `move_ee`, `set_gripper`). LIBERO uses MuJoCo + OSC controllers — direct EE delta control (no Jacobian IK needed).
- `src/sim/libero_success_checker.py`: reads LIBERO's success flag from env info.
- `examples/libero_object.py`: example script wiring LIBERO-Object + VLA, same pattern as `maniskill_pickcube.py`.
- `requirements-libero.txt`: LeRobot + LIBERO deps (**Linux only**; LIBERO requires `sys_platform == 'linux'`).
- `arm_tools.py` must work unchanged with `LiberoArm`.

**Acceptance (manual, on Linux host)**

- `.venv` with LeRobot + LIBERO deps from `requirements-libero.txt`; `export MUJOCO_GL=egl`.
- `python examples/long_horizon_handcrafted_libero.py` with a running VLA server starts a LIBERO-Object task, drives the arm via VLA, and runs to completion (wiring correctness; task success not required).
- On **Windows**: `pytest` passes (LIBERO backend unit tests mock the env; no real LIBERO required).
- `arm_tools.py` works unchanged with `LiberoArm`.

**Colab validation results (Python 3.12, A100 GPU — June 2026)**

- [x] `examples/libero_object.py` wiring (LIBERO-Object task 0 + Pi0.5 VLA): **passed** — 60 arm steps applied, exit 0.
- [x] `pytest` (non-integration, Linux): **160 passed, 2 skipped, 3 deselected**.

Notes: requires A100 (same Pi0.5 weight size as M2). Patches baked into `colab/m3_libero_object.ipynb`: apt `cmake` for `egl_probe` build, lerobot git rev (`0cf8648`), upgraded `transformers`, pre-create `~/.libero/config.yaml` to bypass interactive prompt, `VLA_MAX_TOOL_SECONDS=120` (Pi0.5 JIT warmup on first call), `pi05_libero` checkpoint pre-downloaded via `gsutil` (requires `pip install crcmod` first — Colab's `crcmod` lacks the C extension needed for GCS composite-object integrity checks).

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

- Schema validation accepts well-formed plans and rejects malformed ones with usable errors.
- Interpreter handles every node type (`seq`, `if`, `while`, `call`, `feedback`, `return`) on hand-crafted plans.
- Restricted expression evaluator rejects unsafe input (no attribute access on arbitrary objects, no name lookup outside bound vars, no calls).
- Handcrafted-plan E2E test runs against fully mocked sim + mocked LLM evaluator, exercising both `continue` and `replan` feedback decisions.
- No changes to the VLA HTTP contract.

**Out of scope for M4**

- Real perception, real LLM-emitted plans, real sim integration, recursive sub-planning.

### M5 (implemented ✓)

**Sim-GT skills + handcrafted E2E** on **LIBERO** (Linux primary) and **ManiSkill** (Windows smoke test).

**Scope**

- `src/sim/libero_skills.py`: `perception.detect` / `perception.locate` / `state.list_objects` against LIBERO's MuJoCo scene. `_unwrap_to_sim` walks the LeRobot SyncVectorEnv wrapper chain; skills return `None`/`[]` gracefully if sim is unreachable.
- `src/sim/maniskill_skills.py`: same three skills against ManiSkill scene via `_find_actors` reflection on `env.unwrapped`.
- Sim selection by example import; no config knob.
- `examples/long_horizon_handcrafted_libero.py`: handcrafted plan + `_AlwaysContinue` feedback stub + `_NoReplan` planner, wired to `Executor`.
- `examples/long_horizon_handcrafted_maniskill.py`: same, ManiSkill PickCube — Windows smoke test.

**Acceptance**

- On **Linux**: handcrafted plan completes the chosen LIBERO task end-to-end against LIBERO + a running VLA server.
- On **Windows**: ManiSkill version starts and executes ≥1 skill call without crashing (task success not required; arm calls may fail if no VLA is running).
- `arm.grab` / `arm.place` operate over the unchanged VLA HTTP contract.

**Colab validation results (Python 3.12, A100 GPU — June 2026)**

- [x] `examples/long_horizon_handcrafted_libero.py` wiring (LIBERO-Object task 0 + Pi0.5 VLA): **passed** — 60 arm steps applied, `Executor result: status=done`, exit 0.
- [x] `pytest` (non-integration, Linux): **160 passed, 5 skipped**.

Notes: `grasp_triggered=false` — Pi0.5 (DROID weights) does not reliably grasp in LIBERO sim; wiring correctness only. `obj=?` in VLA step log is expected: `arm.grab` passes a task string to the VLA; object pose from `perception.locate` is not forwarded to the VLA step logger by design.

**Out of scope for M5**

- Real perception (and the `PerceptionBackend` abstraction it motivates), LLM-emitted plans, LIBERO-Long.

### M6 (implemented ✓)

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

- On **Linux + LIBERO-Long**: LLM emits a valid plan from the task language goal, the executor runs it with the VLA in the loop, and the run reaches at least one happy-path completion of the chosen task.
- Schema-violating LLM output triggers a bounded retry with the validation error injected into the next prompt.
- Replan and step-budget caps are enforced.
- On **Windows**: `pytest` passes (256 passed, 2 skipped, 3 deselected).
- Per-run trace file written to `traces/` and contains all expected event types.

**Colab validation results (Python 3.12, A100 GPU — June 2026)**

- [x] LLM-emitted plan (gpt-4o): valid plan generated, executor ran skills, VLA HTTP contract confirmed.
- [x] `LLMFeedbackEvaluator`: RGB snapshot + description sent to LLM, decision returned and respected.
- [x] Trace file written to `traces/` with `plan_start`, `skill_call`, `plan_end` events confirmed.
- [x] `pytest` (non-integration, Linux): **256 passed, 2 skipped, 3 deselected**.
- [x] Happy-path task completion: **LIBERO-10 task 0 — PASS**. `examples/long_horizon_llm_libero.py` exit 0, LIBERO env `is_success=True`, `status=done`, 0 replans, 4 skill steps. Pi0.5 placed the soup can; LIBERO's physics contact detector confirmed task success. LIBERO's internal success gate (contact-based) is the authoritative criterion — the basket body-origin XY distance metric is not used for acceptance.

Notes: bare-primitive args coercion added to `CallNode` schema (`src/core/plan/schema.py`) to handle LLM output where args are plain strings instead of `{"kind": "literal", "value": ...}` expressions. Episode-horizon observability (`env_step_count`, `episode_done` early-abort with `is_success` propagation as `reason=task_success`) added to `LiberoArm`/`vla/tools.py`; `LONGOR_RUN_LOG` scoped file logging added to smoke scripts for diagnostics.

**Post-M6 fix (done ✓)**

- Default `LLM_MODEL` updated from `gpt-4o-mini` → `gpt-4.1-mini` in `.env.example` and validation notebooks. `gpt-4o` and `gpt-4o-mini` are legacy as of 2026; `gpt-4.1-mini` has stronger instruction-following at a comparable price point.
- `LLMFeedbackEvaluator` system prompt reordered to lead with the STOP-semantics constraint: STOP = entire goal done or permanently failed; a positive intermediate sub-goal check = CONTINUE. Root cause: `gpt-4o-mini` called STOP after placing only the first of two objects, terminating the plan early.

**Out of scope for M6**

- Real perception, success across all LIBERO-10 tasks (one task is enough), recursive sub-planning.

### Post-Linux-validation cleanup (done ✓)

Retire the legacy sim-loop path; consolidate tests and notebooks.

**Scope (completed)**

- Deleted `src/sim/runner.py`, `src/sim/tools.py`, `src/sim/policy_stub_tools.py` (sim-loop helpers; no surviving dependents).
- `src/core/llm_agent.py` **stays** — still used by `src/robots/XLeRobot/` and `src/robots/EarthRover/` HW agents.
- Deleted `examples/libero_object.py` and `examples/maniskill_pickcube.py`; M1/M3 acceptance criteria updated to reference `long_horizon_handcrafted_maniskill.py` / `long_horizon_handcrafted_libero.py`.
- Deleted legacy test files (`test_e2e_sim_loop.py`, `test_sim_loop.py`, `test_sim_runner.py`, `test_policy_stub_tool.py`); merged overlapping test files (`test_arm_tools` → `test_maniskill_arm_adapter`, `test_e2e_handcrafted_plan` → `test_plan_executor`, `test_vla_adaptive_scaling` → `test_vla_action`, `test_sim_tools_contract` → `test_maniskill_backend`).
- Notebooks reduced from 5 to 2: deleted `m1_maniskill_octo.ipynb`, `m2_pi05_vla.ipynb`, `m3_libero_object.ipynb`; compacted `m5_handcrafted_libero.ipynb` and `m6_llm_libero.ipynb`.

---

### M7 (optional, later)

**Real perception backend.** Open-vocab detector (e.g. Grounding DINO or OWL-ViT + SAM2) implementing the same `perception.`* skill set as the sim-GT versions, but against rendered RGB instead of sim ground truth — so a single sim can be paired with either sim-GT or real perception. Introduces a `PerceptionBackend` interface in `src/core/skills.py` and a config knob (e.g. `PERCEPTION_BACKEND=sim_gt | openvocab`). Linux + GPU. Out of current planning scope; recorded so the option is on file.

**M1–M3** widen OS, sim, and policy options; **M4–M6** add the planner architecture on top, with **LIBERO** as the primary E2E target and **ManiSkill on Windows** as the cross-platform smoke test.

## Validation tooling (Colab)

Linux milestone acceptance runs via notebooks in `colab/`, one per milestone (`colab/mN_validate.ipynb`). Each notebook is self-contained and ephemeral-safe: no Google Drive dependency; the repo is cloned and deps installed into the Colab VM's local disk on each session. The GitHub PAT and any API keys are read from Colab Secrets (never embedded in notebook cells). `.venv` is created with `--system-site-packages` to inherit Colab's pre-installed CUDA PyTorch; `.vla-venv` is fully isolated. The VLA server runs as a background subprocess; the notebook polls `/health` before proceeding.

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
- **Simulation and hardware** live in `**sim/`**, `robots/**`,** or dedicated backend modules; select backend via **config/env**.
- **Optional** heavy or HW-only dependencies: **lazy imports** and/or `requirements-optional.txt` so minimal sim dev does not install everything.
- **Plan representation:** structured JSON AST (schema in `src/core/plan/schema.py`); the executor walks it with a restricted expression evaluator. **No `exec`** of generated code; **no `eval`** of arbitrary expressions.
- **Skills:** `Skill` base + `SkillRegistry` live in `src/core/skills.py` (interfaces only). Concrete skill implementations live in their backend packages (`src/sim/*_skills.py`, `src/vla/arm_skills.py`, future `src/robots/*_skills.py`) and self-register at import time.
- Adding a new sim, robot, or capability: add a backend adapter and skill implementations in the relevant backend package; the planner / executor stay unchanged.
- **`core/llm_agent.py`** — the general-purpose LLM tool-call loop used by HW robot agents (`XLeRobot`, `EarthRover`). Not a sim concern; retained as-is.

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
- `**pytest.ini`** sets `pythonpath = src`.

