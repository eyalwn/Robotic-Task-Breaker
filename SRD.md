# Longor SRD

Single source of truth for Longor runtime architecture and contracts.
Operational setup/run instructions intentionally live in `README.md` and
notebook launchers under `colab/`.

## Goal

Longor separates high-level reasoning from low-level control for long-horizon
robot manipulation tasks.

- `LLMPlanner` turns a natural-language goal and RGB snapshot into a validated
  plan AST.
- Plans are declarative data (`plan.py`), not executable Python.
- `Executor` walks the AST, executes skills, and uses the VLA HTTP service for
  arm manipulation.
- `Runner` runs one observe → plan → execute pass per episode until success,
  failure, planner STOP, or episode step budget.
- The planner may emit a sentinel STOP plan (`"STOP: <reason>"`) when
  continuation is not useful.

## Platform

Supported target: **LIBERO on Linux with NVIDIA/CUDA** (Colab T4 is sufficient).
Runtime assumes one environment backend (LIBERO) and one policy transport
(single VLA HTTP service).

## Milestones

Only runtime accomplishments are milestones. Supporting refactors, harness code,
and bug fixes are tasks inside a milestone, never milestones of their own.

- **M1 — DONE. Closed-loop VLA control solves a LIBERO task.**
  `validation/libero_probe.py`, then still carrying sim ground-truth diagnostics,
  drove the policy against `libero_object` task 0 with no planner, executor,
  skills, or plan AST, and reached the BDDL success predicate. Establishes that
  the env wrapper, policy service, and checkpoint are correctly wired.

- **M2 — DONE. Monolithic baseline success rate on `libero_10` task 0.**
  N episodes over distinct initial states, single direct task prompt, no planner.
  Measured on our stack (10 episodes each):
  - `VLA_ACTIONS_PER_CHUNK=1`: `3/10` (30%)
  - `VLA_ACTIONS_PER_CHUNK=5`: `6/10` (60%)
  - `VLA_ACTIONS_PER_CHUNK=10`: `4/10` (40%)
  Selected baseline setting: **`VLA_ACTIONS_PER_CHUNK=5`**.
  This is the control arm for every later comparison, so it is measured on our
  own stack rather than taken from published numbers. For reference, SmolVLA
  0.45B reports 71% on LIBERO-Long, while independent reproductions of the
  `HuggingFaceVLA/smolvla_libero` checkpoint report 38-56%.

- **M3 — Longor solves `libero_10` task 0 end to end.**
  LLM plan, decomposed arm skills, real perception, at least one episode
  reaching the BDDL success predicate.

- **M4 — Longor beats the M2 baseline on `libero_10` task 0.**
  Same N, same initial states, same episode budget, both arms. Acceptance:
  Longor's success rate exceeds the monolithic baseline's.

- **M5 — Full LIBERO-10 campaign.**
  All 10 tasks x N episodes for both arms, resumable across sessions, reported
  as a per-task table against published SmolVLA numbers.
  Three known obstacles, all specific to widening past task 0:
  task 8 ("put both moka pots on the stove") has `moka_pot_1` and `moka_pot_2`,
  which normalise to the same name and cannot be told apart by conjunct
  matching; task 2 needs a `Turnon` conjunct but the skill catalog has no
  turn-on/toggle skill, which is a capability gap rather than a naming one; and
  task 3's goal names `white_cabinet_1_bottom_region` while its language says
  "bottom drawer", so planner wording will not match.

## Architecture

Source layout is flat modules under `src/`. Dependencies flow downward only:

`config -> libero_env / vla_client -> plan -> skills -> executor -> planner -> runner`

Module responsibilities:

- `config.py`: single source of runtime defaults and `.env` switches.
- `libero_env.py`: LIBERO wrapper for `observe`, `step`, `success`, subtask
  completion, initial-state selection, and episode step accounting
  (`steps` / `max_steps`). The only module permitted to touch sim ground truth.
- `vla_client.py`: HTTP client for the VLA policy service.
- `plan.py`: plan AST models, validation, and STOP helpers.
- `skills.py`: skill interfaces/registry and static arm-skill catalog text.
- `executor.py`: plan interpreter; owns VLA-to-environment execution for
  `arm.grab`, `arm.place`, `arm.transfer`, `arm.open`, `arm.close`,
  `arm.push`, and `arm.pull`.
- `planner.py`: LLM planner that emits validated AST plans from a goal and
  snapshot (no sim ground-truth object injection; no feedback nodes).
- `runner.py`: composition root for single-pass observe → plan → execute per task.

Non-negotiable boundaries:

- VLA-to-environment integration lives in `executor.py` in one visible path.
- No hidden retuning layer between policy output and `env.step()` (no remap,
  rescale, sign flips, or success rewriting).
- `config.py` is the only place runtime defaults are defined.
- No sim ground truth in action selection or object naming. The planner names
  objects from the RGB snapshot and the natural-language goal; the policy sees
  only camera frames and proprioception.
- **Subtask termination is the one documented exception.** `LiberoEnv` exposes a
  subtask-completion check backed by LIBERO's own goal predicates, and
  `executor.py` uses it to end an arm skill as soon as its subtask is satisfied.
  This is deliberate research scaffolding on two grounds: it isolates the
  decomposition hypothesis from the separate problem of *perceiving* subtask
  completion, and a perfect oracle upper-bounds the benefit decomposition can
  deliver, so a null result under it kills the hypothesis cheaply. Results
  produced with it must be reported as "given a subtask-completion oracle".
  Replacing it with a perception-based verifier is Future Work.

## Episode budget and arm-skill contract

- Episode hard cap is suite `MAX_EPISODE_STEPS` via `LiberoEnv.max_steps`
  (e.g. `libero_10` = 520). Budgeting counts real `env.step()` calls.
- Per-skill cap is `MAX_SKILL_STEPS` = 120, also clamped by remaining episode
  budget. Sized so a typical four-skill LIBERO-10 plan fits the 520-step episode
  cap (520 / 4 = 130). With subtask termination in place this cap is a guard
  against a subtask that is failing, not a per-stage schedule, so erring loose is
  safer than erring tight: a tight cap severs a healthy skill mid-motion, while a
  loose one costs nothing on the success path.
- An executor-handled arm skill ends when its subtask predicate is satisfied, the
  global task succeeds, the per-skill cap is reached, or the episode budget runs
  out. Normal completion returns:
  `{"status": "done" | "timeout", "env_steps": <int>, "subtask_success": <bool>,
  "task_success": <bool>}`
  so logs identify which stage failed. `task_success` remains the global LIBERO
  BDDL predicate. Execution exceptions surface as `StopReason.EXECUTION_ERROR`,
  not as a fake skill failure status.

### Subtask termination mechanism

Two signals, because LIBERO has no BDDL grasp predicate:

- **Placement-style skills** (`arm.place`, `arm.transfer`, `arm.open`,
  `arm.close`, `arm.push`, `arm.pull`) resolve to one goal conjunct in
  `parsed_problem["goal_state"]`, evaluated via the problem class's
  `_eval_predicate`.
- **`arm.grab`** has no conjunct — "pick up" is never a LIBERO goal condition —
  so it uses robosuite's `_check_grasp` against the object's contact geoms.

Conjunct resolution matches a skill's primary object against each conjunct's
first argument under the same normalisation the executor already applies to
build VLA task strings (strip a trailing `_<n>`, underscores to spaces):
1. exact normalised equality;
2. if none, bidirectional containment (`a in b` or `b in a`).
If exactly one conjunct matches, it is the subtask predicate. **If zero or
several match, the skill falls back to the step-budget cap and logs a warning**
— an unresolvable name must not abort the episode, because it indicates a
planner naming mismatch rather than a manipulation failure.

`Runner(task_suite, task_idx)` always plans from `LiberoEnv.task_description`
(no hand-written goal argument).

## Evaluation protocol

- Episodes vary the LIBERO initial state: episode `i` uses
  `get_task_init_states(task_idx)[i]`. Repeating a single initial state measures
  planner variance, not task success rate, and long-horizon suites are the most
  sensitive to this because error compounds across stages.
- Comparison arms must match on everything but the variable under test: same task
  string (`LiberoEnv.task_description`, never a hand-written goal), same initial
  states, same episode budget, same `VLA_ACTIONS_PER_CHUNK`.
- `VLA_ACTIONS_PER_CHUNK` is LeRobot's `n_action_steps`: how many actions of each predicted
  chunk are executed before re-querying the policy. It is known to swing LIBERO-10
  results substantially and is therefore a measured parameter, not a free one.

## Configuration Contract

`.env` is intentionally minimal: `OPENAI_API_KEY`, `LOG_LEVEL`, `LLM_MODEL`,
`SAVE_VIDEO`.
All other tunables live in `src/config.py`.
Adding a new env var requires explicit SRD-level approval.

## VLA Service Contract

Single policy service process (`services/smolvla_service.py`) exposes:

- `GET /health` -> `{"status":"ok","model":"<id>"}`
- `POST /predict` (multipart):
  - `image` (JPEG), `image_wrist` (JPEG), `task` (string),
    `state` (JSON array of 8 floats)
  - response: `{"actions": [[7 floats], ...]}`

Contract changes must be recorded here before implementation changes.

## Future Work

- Replace the sim ground-truth subtask oracle with a perception-based verifier,
  so the system is deployable outside simulation.
- Revisit recursive sub-planning only after baseline stability is demonstrated.

Longer-range goals, out of scope until LIBERO-10 is settled:

- **Mobile manipulation.** Extend beyond a fixed-base arm to a wheeled robot,
  with a benchmark that exercises navigation together with manipulation. This
  adds a second skill family alongside `arm.*` and is the first real test of
  whether the plan AST generalises past one embodiment.
- **Real hardware.** Run Longor on a physical XLeRobot.
