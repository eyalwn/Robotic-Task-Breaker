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
- **M2 — DONE.** Historical Longor OpenDrawer run on RoboCasa with
  GR00T N1.6-3B through the HTTP VLA contract succeeded in one `Runner` episode.
  Report as "given a subtask-completion oracle."
- **M3 — DONE (drawer open->close chain).** Task: OpenDrawer scene,
  goal `<reset language>, then close it` (e.g. `open the left drawer, then close
  it`), plan `open(<side> drawer)` -> `close(<side> drawer)`. Selection rule
  (learned from the Kettle/SnackSorting failures): a Longor primitive is
  admissible only if its exact prompt wording is in the checkpoint's training
  set. `nvidia/GR00T-N1.6-3B` was trained on 24 RoboCasa atomic tasks
  (`single_panda_gripper.*`; the "66.22%" NVIDIA figure is the average over
  exactly those, each in its own scene), so unseen composite tasks and
  unseen prompt wording are expected to fail. Training prompts: `open the
  left|right drawer`, `close the left|right drawer`. The scene is the same
  `ManipulateDrawer` class as upstream CloseDrawer, whose start state (door
  90-100% open) matches the post-`open` state. Gates (GR00T N1.6-3B,
  PandaOmron observations/actions, 8 actions/call, backend horizon 1000, 10
  episodes each, threshold **>=7/10**): (1) in-scene chain gate, fixed plan,
  no LLM, reports `open_ok`, `close_ok_given_open`, `success`; (2) only if (1)
  passes: bare arm (task description as one prompt) vs Longor `Runner`,
  reported as "given a subtask-completion oracle." Episode `success()` =
  drawer closed now AND opened earlier in the episode (latched), because
  OpenDrawer starts closed. This is a chaining/usefulness check on a single
  task, not the M5 claim. Observed run artifact
  `G:\My Drive\Colab\longor_m3_kettle_bare\20261015T094943Z\drawer_chain_gate.log`
  reports `open_ok=9/10`, `close_ok_given_open=7/9` (77.8%), and
  `success=7/10`, so gate (1) passes its threshold.
  Failure handling remains strict: ep4 and ep6 stay failed
  (`open_ok=True`, `close_ok=False`, `success=False`) and ep8 stays failed
  (`open_ok=False`, close not attempted). Ep4's video appearance does not
  override the sim-GT oracle result, and ep8 target/framing ambiguity was not
  excluded post hoc.
  Rejected under this checkpoint contract (history, code removed):
  `KettleBoiling` bare-VLA 0/10, in-scene place gate 0/5 (stove-burner
  destination wording is out of distribution; the burner phrase triggered the
  TurnOnStove behavior); `SnackSorting` in-scene `put_in` gate **0/10**
  (`pick the bar from the drawer and place it in the bowl` is not a training
  phrase: drawer is never a PnP source, `in the bowl` only appears with a pan
  source; runs hovered at the drawer or pushed the bowl).
  Reliable trained primitives for future suites (upstream per-task rate,
  >=85%): CloseDrawer 100, CoffeePressButton 98.5, CloseSingleDoor 96,
  TurnOffMicrowave 96, TurnOffSinkFaucet 93.5, TurnOnMicrowave 91.5,
  TurnOnSinkFaucet 89, CloseDoubleDoor 88.5, TurnSinkSpout 87. Pick-and-place
  tasks are 19-63% upstream.
- **M4 — IN PROGRESS.** Longor-vs-bare-VLA pilot on the M3 suite. Reset scenes
  are unpaired until reproducibility is resolved, so pilot rates are diagnostic
  only. The initial artifact
  `G:\My Drive\Colab\longor_m4_pilot\20261015T181402Z` observed bare `0/10` and
  Longor `4/10`; arms saw independently generated scenes, so it does not support
  a win/loss claim. Retain per-episode artifacts and assign every failure to
  exactly one category: policy/primitive, planner plan, plan-to-skill grounding,
  subtask transition/oracle, or infrastructure. Use the pilot to freeze the M5
  suite, budgets, no-replanning policy, sample size, and decision rule. Before
  the
  next batch, use the single-run `open_close_drawer` debug launcher to validate
  stage/oracle behavior; then freeze an episode eligibility and observability
  rule (including ambiguous/framing edge-cases) for both arms. Batch launcher:
  `validation/run_m4_pilot.py` (artifact outputs:
  `m4_pilot_episodes.jsonl`, `m4_pilot_summary.json`, per-attempt Longor
  records, and optional videos).
  **Frozen M4/M5 runtime protocol:** Longor scored episodes use a single
  observe→plan→execute pass (no timeout-triggered suffix retry/replan). The only
  default hard step cap is the backend episode horizon (`env.horizon`: 1000 for
  `open_close_drawer`). `config.MAX_SKILL_STEPS` (`open` 720, `close` 400) is
  observational telemetry only (threshold crossings captured in executor
  telemetry and persisted in pilot artifacts); it is not a default runtime stop.
  Final success is always `env.success()` at the terminal state. The pre-freeze
  unpaired artifact
  `G:\My Drive\Colab\longor_m4_pilot\20261015T181402Z` remains diagnostic-only
  and does not support formal win/loss claims. M3 drawer-chain gate behavior
  stays separate: it enforces per-skill caps via explicit
  `Executor.run(..., skill_step_limits=...)` (not global config mutation). A
  close-to-horizon debug probe (`run_drawer_chain_gate.py --close-to-horizon`)
  inspects oracle/joint trajectories and is not counted as M4 evaluation output.
- **M5 — PLANNED.** Formal Longor-vs-bare-VLA comparison. Freeze code, model
  checkpoint, prompts, task IDs/seeds, action chunking, compute limits, and
  stop/timeout policy before execution. Both arms use the same VLA server,
  observation adapter, action representation, horizons, and seeds; the bare
  arm receives original task language while Longor receives generated primitive
  prompts. Report per-episode paired outcomes, success rates with uncertainty,
  and simulation-ground-truth use for subtask checks. The predeclared decision
  rule requires the lower confidence bound of Longor's absolute success-rate
  lift across the suite to be positive. Pairing requires reproducible scenes:
  `reset(seed=...)` through the GR00T wrapper was verified NOT reproducible
  (same seed -> different frames); resolve (seed hook vs unpaired comparison)
  before M4 freezes the protocol.

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
  `steps`, `horizon`, and `task_description`.
- Current backend slice: `src/envs/robocasa_env.py` supports
  `task in {"open_drawer", "open_close_drawer"}` and `task_idx=0` only.
- `observe()` returns `(images, state)` where `images` is a tuple of lossless
  PNG bytes and `state` is a float list. For RoboCasa today: 3 images.
- VLA integration lives in `src/executor.py`; actions from `vla_client` are
  passed to `env.step(action)` unchanged (shape/finiteness validation only).
- `Runner` owns planner STOP handling and the final task verdict; `Executor`
  only reports execution outcomes.
- `plan(goal, snapshot)` takes the goal (`env.task_description`) per call.
- RoboCasa backend: robosuite render contexts are finalised in their own EGL
  context (`_free_render_contexts_in_own_egl_context`). An old sim survives
  `reset()` as cyclic garbage; when the cyclic GC finalises it, stock
  `MjRenderContext.__del__` frees GL objects in the live context and every later
  camera frame is noise or black (Colab probe, Oct 2026: GC forced after init
  corrupted 5/5 runs; finalising in the owning context was clean 5/5).
- Sim ground truth is allowed only for subtask/episode completion checks
  (`check_subtask`, `success`) and must be disclosed in evaluation claims.
- Unsupported backend API is a hard stop with `'not supported for this env'`.

## Plan and skill contract

Planner output must be valid JSON in one of two shapes:

- Step array: `[{"skill":"open","object_name":"..."}]`
- STOP sentinel: `{"stop":"<reason>"}`

`open` and `close` take `object_name`. The planner decides all names before
execution (one plan from the initial snapshot) and must be specific whenever a
name could match more than one object (left/right drawer); the same object is
named identically across steps.

Runtime skill surface (prompts are exact GR00T training wording):

- `open(object_name)` -> VLA task prompt `open the <object_name>`.
- `close(object_name)` -> VLA task prompt `close the <object_name>`.

Subtask oracles (`check_subtask`, sim ground truth, disclosed in claims):
`open` = robosuite OpenDrawer `_check_success` (drawer >= 95% open); `close` =
every normalized drawer door joint `<= 0.05`.

Per-skill telemetry thresholds (`config.MAX_SKILL_STEPS`): `open` 720, `close`
400. These are logged when crossed during a skill; they do not stop default
Longor runtime (M4/M5). Optional hard per-skill caps are passed explicitly as
`Executor.run(..., skill_step_limits=...)` (M3 drawer-chain gate and debug
probes only).

Plan-once semantics: `Runner` and M4 Longor episodes call the planner once
from the initial snapshot, execute the full plan once, and do not retry an
unfinished suffix after timeout. Genuine replanning remains out of scope.

Terminal verdict semantics: final task success is `env.success()` at the final
state, regardless of whether execution terminated via `plan_done` or
`episode_timeout`. Stop reason remains a diagnostic field and is logged
separately from the verdict.

Episode step cap comes from the backend (`env.horizon`: 720 for `open_drawer`,
1000 for `open_close_drawer`), not from a global config constant.

## M3 drawer-chain gate (completed)

In-scene fixed plan, no LLM (`validation/run_drawer_chain_gate.py`): step 1
`open(<side> drawer)`; step 2 `close(<side> drawer)` only if step 1 succeeds,
with `<side>` parsed from the reset language. Ten episodes default;
per-episode unique video path when `SAVE_VIDEO=1`; simulator ground truth for
stage oracles. Viability threshold: **>=7/10** per primitive.
Observed run (artifact
`G:\My Drive\Colab\longor_m3_kettle_bare\20261015T094943Z\drawer_chain_gate.log`)
met the gate with `open_ok=9/10`, `close_ok_given_open=7/9`, and `success=7/10`.

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
