# Longor SRD

Architecture, milestone, and runtime-service contracts for Longor. Setup and
runbook instructions live in `README.md`.

## Hypothesis

Longor beats a bare VLA on long-horizon tasks.

This hypothesis is platform-agnostic. Milestones pick a concrete
simulator/checkpoint pair only to test that claim.

## Milestones

- **M1 — DONE.** Upstream-only RoboCasa / GR00T N1.6 OpenDrawer gate
  (`colab/run_robocasa_primitive_gate.ipynb` ->
  `validation/run_robocasa_primitive_gate.py`): pass threshold = >=7/10 episodes;
  observed 8/10 (10 episodes, 720-step cap, 8 actions/call). Artifacts are
  external (Drive), not committed.
- **M2 — DONE.** Longor OpenDrawer on RoboCasa with GR00T N1.6-3B through the
  HTTP VLA contract (`colab/run_longor_task.ipynb` -> `examples/run_longor_task.py`):
  succeeded in one `Runner` episode. Report as "given a subtask-completion oracle."
- **M3 — PLANNED.** Long-horizon capability gate: select a small RoboCasa suite
  requiring at least two independently verifiable skills; extend the environment,
  skill catalog, planner schema, and VLA prompt mapping only for its primitives;
  then run a matched bare-VLA primitive gate for every required primitive.
  The gate uses the planned comparison's model, observation/action contract,
  action chunking, episode cap, and task/seed distribution. A primitive that is
  not viable makes a Longor comparison result uninterpretable as a planning result.
- **M4 — PLANNED.** Paired Longor-vs-bare-VLA pilot on the M3 suite. Retain
  per-episode artifacts and assign every failure to exactly one category:
  policy/primitive, planner plan, plan-to-skill grounding, subtask
  transition/oracle, or infrastructure. Use the pilot to freeze the M5 suite,
  budgets, replanning policy, sample size, and decision rule; do not claim a
  win or loss from it.
- **M5 — PLANNED.** Formal Longor-vs-bare-VLA comparison. Freeze code, model
  checkpoint, prompts, task IDs/seeds, action chunking, compute limits, and
  stop/timeout policy before execution. Both arms use the same VLA server,
  observation adapter, action representation, horizons, and seeds; the bare
  arm receives original task language while Longor receives generated primitive
  prompts. Report per-episode paired outcomes, success rates with uncertainty,
  and simulation-ground-truth use for subtask checks. The predeclared decision
  rule requires the lower confidence bound of Longor's absolute success-rate
  lift across the suite to be positive.

## Runtime architecture contract

Source layout is flat modules under `src/`, with sim backends under
`src/envs/` only.

Dependency order (strictly downward):

`config -> vla_client / envs.sim_env -> plan -> skills -> executor -> planner -> runner`

Contract points:

- `src/config.py` is the only runtime reader of `os.environ` and the only
  defaults owner.
- `SimEnv(task, task_idx=0)` lazy-imports the backend into `self._env` and
  forwards `observe`, `step`, `success`, `check_subtask`, `close`, plus
  `steps` and `task_description`.
- Current backend slice: `src/envs/robocasa_env.py` supports `task="open_drawer"`
  and `task_idx=0` only.
- `observe()` returns `(images, state)` where `images` is a tuple of lossless
  PNG bytes and `state` is a float list. For RoboCasa today: 3 images.
- VLA integration lives in `src/executor.py`; actions from `vla_client` are
  passed to `env.step(action)` unchanged (shape/finiteness validation only).
- `Runner` owns planner STOP handling and the final task verdict; `Executor`
  only reports execution outcomes.
- Sim ground truth is allowed only for subtask/episode completion checks
  (`check_subtask`, `success`) and must be disclosed in evaluation claims.
- Unsupported backend API is a hard stop with `'not supported for this env'`.

## Plan and skill contract

Planner output must be valid JSON in one of two shapes:

- Step array: `[{"skill":"open","target_name":"..."}]`
- STOP sentinel: `{"stop":"<reason>"}`

Runtime skill surface:

- `open(target_name)` -> VLA task prompt `open the <target_name>`.

## VLA service contract

Launch with `python services/run_vla_server.py`.
`config.VLA_POLICY` selects the service (current default/implementation: `groot`,
bridging HTTP `:8765` to upstream GR00T ZMQ `:5555`).

- `GET /health` -> `{"status":"ok","model":"<id>"}` with optional integer
  `action_dim`.
- `POST /predict` multipart:
  - `images`: PNG parts named `images` (order matches `observe()`)
  - `task`: string
  - `state`: JSON array of floats
  - response: `{"actions": [[...], ...]}`
- Service enforces `config.VLA_ACTIONS_PER_CHUNK` as a maximum actions-per-response.
- Raw action guarantee: validate shape and finiteness only; no clipping,
  rescaling, remapping, or sign changes.
- Backend-specific camera/state mismatches must return HTTP 400 (no silent
  padding/dropping).

`src/vla_client.py` remains backend-agnostic. Contract changes are recorded here
before implementation changes.

## Historical findings (legacy stacks, concise)

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
