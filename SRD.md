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
  `success=7/10` in the in-scene gate.
- **M4 - DONE.** Unpaired RoboCasa `open_close_drawer` comparison: Longor
  `16/20` (80%) vs bare VLA `2/20` (10%), one-sided Fisher exact
  `p=8.32e-6`. This supports the claim only for the evaluated task distribution.
  Recorded at `826b8e5`. Since then the planner's JSON-repair retries were
  removed (`003ccca`) and the re-plan loop was restored (`786f8c5`); the loop
  does not trigger on the M4 tasks (`Executor` reads `config.MAX_SKILL_STEPS`,
  which sits above the episode horizons, so no subtask timeout can occur).

## Runtime Architecture Contract

- Source layout is flat modules under `src/`, with sim backends only under
  `src/envs/`.
- Dependency direction is strict:
  `config -> vla_client/envs.sim_env -> plan -> skills -> executor -> planner -> runner`.
- Module responsibilities (rough; each module's docstring is the detailed
  description and must agree with this list):
  - `config`: single source of truth for configuration and the only runtime
    reader of `os.environ`.
  - `vla_client`: HTTP client for the VLA policy service.
  - `envs.sim_env`: thin facade; builds and forwards to the sim backend
    selected by `config.SIM_ENV`.
  - `envs.robocasa_env`: the RoboCasa backend (observations, actions, and
    sim-ground-truth subtask/goal checks).
  - `plan`: plan types, validation, and the STOP sentinel.
  - `skills`: skill catalog shown to the planner.
  - `executor`: runs a plan's steps, turning each skill into a VLA prompt and
    stepping the env with the VLA's actions until the subtask check passes.
  - `planner`: LLM planner; goal plus scene snapshot in, plan out.
  - `runner`: wires the above for one task and runs observe -> plan ->
    execute (re-plan after a subtask timeout); the final verdict is always
    `env.success()`.
  - `utils`: logging setup for process entrypoints.

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

## Historical Findings (concise)

Legacy LIBERO/VLABench stacks and rejected RoboCasa tasks were removed from the
current runtime. Findings remain as milestone context only.

- **RoboCasa subtask-selection rule (`9a711b7` SRD):** a subtask is admissible
  only if its exact prompt wording is in the checkpoint's training set.
  `nvidia/GR00T-N1.6-3B` was trained on 24 atomic RoboCasa tasks, including
  `open|close the left|right drawer`; composite tasks and unseen wording are
  expected to fail. The M4 composite prompt (`<open phrase>, then close it`)
  is not a training prompt.
- **RoboCasa rejected tasks (`637a88f`, `4921d1d`, `9a711b7`):** `KettleBoiling`
  bare VLA 0/10 and in-scene place gate 0/5 (burner wording triggered
  `TurnOnStove` behavior); `SnackSorting` in-scene `put_in` gate 0/10
  (`pick the bar from the drawer and place it in the bowl` is not a training
  phrase; runs hovered at the drawer or pushed the bowl).

- **Pi05 / LIBERO subtask-language finding (`19d1c585` era SRD):**
  `pi05_libero` mostly ignored subtask wording in-scene (e.g., asked for tomato
  sauce first, still approached alphabet soup). Qualitative finding; not
  reproducible from the current tree.
- **Pi05 / LIBERO E2E milestone (`9b437ef`, `6641b39`):**
  one LIBERO-10 task-0 Longor run reached `is_success=True` / `status=done`,
  and M1-for-pi05 was marked done. Evidence is commit history + external Colab
  artifacts.
- **SmolVLA / LIBERO prompt sweep (2026-08-14..15, `8fac337` notebook):**
  on `libero_10` task 0 the first subtask (`pick up the alphabet soup`) failed
  in 4/4 Colab runs. A sweep scored with the `grab` subtask check grasped for
  2/5 prompts at both 120- and 280-step budgets: the original task
  description and `put the alphabet soup in the basket`. `pick up the
  alphabet soup`, `pick up the alphabet soup and place it in the basket`, and
  `grasp the alphabet soup` failed, so the step budget was not the cause. Raw
  notebook output was not committed.
- **LIBERO memorization hypothesis (2026-08-17 chats):** a code review found
  no path by which the original task text reaches the VLA. The behavior (SmolVLA
  wording sensitivity above; Pi05 following the original task order) fits a
  VLA fitted to fixed scene/instruction pairs. The control that would confirm
  it (an empty or unrelated in-distribution prompt in the same scene) was
  proposed but not run, so this stays a hypothesis.
- **SmolVLA gate failure (`d2046b1`, `c42de0e`):**
  `lerobot/smolvla_vlabench` on VLABench `select_fruit` ended 0/8 before OOM;
  not diagnostic for a Longor comparison.
- **Legacy VLABench rates (`19d1c585`, `c42de0e`):**
  bare `pi0-fast` baseline reached 6/10 on held-out `select_fruit`; Longor on
  `cook_dishes` reached 0/10 (6 physical stalls, 4 oracle-grounding mismatch).
  Raw run artifacts are external.
