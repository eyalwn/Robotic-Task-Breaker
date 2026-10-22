# Longor SRD

Architecture and evaluation contract for Longor. Setup and runbook instructions
live in `README.md`.

## Hypothesis

Longor beats a bare VLA on long-horizon tasks.

## Scope

- Current runtime backend: RoboCasa (`src/envs/robocasa_env.py`) only.
- Supported tasks: `open_drawer`, `open_close_drawer` with `task_idx=0`.
- Sim ground truth is allowed only for subtask/episode completion checks.

## Milestones

- **M1 - DONE.** Upstream-only RoboCasa / GR00T N1.6 OpenDrawer gate passed
  (`8/10` with threshold `>=7/10`).
- **M2 - DONE.** Historical single Longor OpenDrawer run succeeded in one
  `Runner` episode ("given a subtask-completion oracle").
- **M3 - DONE.** Drawer open->close chain on `open_close_drawer` reached
  `success=7/10` in the in-scene gate (`validation/run_drawer_chain_gate.py`).
- **M4 - DONE.** Unpaired RoboCasa `open_close_drawer` comparison: Longor
  `16/20` (80%) vs bare VLA `2/20` (10%), one-sided Fisher exact
  `p=8.32e-6`. This supports the claim only for the evaluated task distribution.

## Runtime Architecture Contract

- Source layout is flat modules under `src/`, with sim backends only under
  `src/envs/`.
- Dependency direction is strict:
  `config -> vla_client/envs.sim_env -> plan -> skills -> executor -> planner -> runner`.
- `src/config.py` is the only runtime reader of `os.environ`.
- `SimEnv(task, task_idx=0)` owns backend instantiation and forwards
  `observe`, `step`, `success`, `check_subtask`, `close`, `steps`,
  `horizon`, `task_description`.
- `observe()` returns `(images, state)` where `images` is 3 lossless PNG frames
  and `state` is a flattened float list.
- VLA integration lives in `src/executor.py`; actions from `src/vla_client.py`
  are passed to `env.step(action)` unchanged aside from shape/finiteness
  validation.
- Planner output is either:
  - step array: `[{"skill":"open","object_name":"..."}, ...]`, or
  - STOP sentinel: `{"stop":"<reason>"}`.
- Runtime skill prompts stay exact:
  - `open(object_name)` -> `open the <object_name>`
  - `close(object_name)` -> `close the <object_name>`
- Episode horizons are backend-owned: `open_drawer=720`,
  `open_close_drawer=1000`.
- `config.MAX_SKILL_STEPS` (`open=720`, `close=400`) is telemetry-only in the
  default runtime; it does not stop execution by itself.
- Optional hard per-skill caps are passed explicitly via
  `Executor.run(..., skill_step_limits=...)`.
- Plan-once semantics: one planner call from the initial snapshot, then one
  execution pass; no suffix retry/replan in runtime.
- Subtask oracles are sim-ground-truth checks:
  - `open`: RoboCasa OpenDrawer `_check_success` (drawer >= 95% open)
  - `close`: every normalized drawer-door joint `<= 0.05`
- RoboCasa rendering contract: render contexts are finalized in their owning
  EGL context to avoid post-GC frame corruption.
- Terminal verdict is always `env.success()` at the final state.

## VLA Service Contract

Launch with `python services/run_vla_server.py`.
`config.VLA_POLICY` selects the service (current implementation: `groot`,
HTTP `:8765` to upstream GR00T ZMQ `:5555`).

- `GET /health` returns `{"status":"ok","model":"<id>"}` and may include
  integer `action_dim`.
- `POST /predict` multipart:
  - `images`: PNG parts named `images` (order matches `observe()`)
  - `task`: string
  - `state`: JSON array of floats
  - response: `{"actions": [[...], ...]}`
- Service enforces `config.VLA_ACTIONS_PER_CHUNK` as the max actions per
  response.
- Raw action guarantee: shape/finiteness validation only; no clipping,
  remapping, rescaling, or sign changes.
- Backend-specific camera/state mismatches must fail as HTTP 400.

## Historical Findings (legacy stacks, concise)

Legacy LIBERO/VLABench stacks were removed from the current runtime. Findings
remain as milestone context only.

- **Pi05 / LIBERO subtask-language finding (`19d1c585` era SRD):**
  `pi05_libero` mostly ignored subtask wording in-scene (e.g., asked for tomato
  sauce first, still approached alphabet soup). Qualitative finding; not
  reproducible from the current tree.
- **Pi05 / LIBERO E2E milestone (`9b437ef`, `6641b39`):**
  one LIBERO-10 task-0 Longor run reached `is_success=True` / `status=done`,
  and M1-for-pi05 was marked done. Evidence is commit history + external Colab
  artifacts.
- **SmolVLA gate failure (`d2046b1`, `c42de0e`):**
  `lerobot/smolvla_vlabench` on VLABench `select_fruit` ended 0/8 before OOM;
  not diagnostic for a Longor comparison.
- **Legacy VLABench rates (`19d1c585`, `c42de0e`):**
  bare `pi0-fast` baseline reached 6/10 on held-out `select_fruit`; Longor on
  `cook_dishes` reached 0/10 (6 physical stalls, 4 oracle-grounding mismatch).
  Raw run artifacts are external.
