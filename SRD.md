# Longor SRD

Single source of truth for Longor runtime architecture and contracts.
Operational runbook details live in `README.md`.

## Goal

Longor keeps a minimal split between planning and control:

- `LLMPlanner` emits either:
  - an executable flat step array (`list[dict[str, str]]`), or
  - a STOP sentinel object (`{"stop":"<reason>"}`).
- `Executor` executes the given step list exactly as provided.
- `Runner` owns STOP handling and final task verdict (`env.success()`).

This is intentionally not an AST system.

## Platform

Supported target: **LIBERO on Linux with NVIDIA/CUDA** (Colab T4 is sufficient).
Runtime assumes one environment backend (LIBERO) and one policy transport
(single VLA HTTP service).

## Milestones

Only runtime accomplishments are milestones. Supporting refactors and harness
code are tasks inside milestones.

- **M1.** Closed-loop VLA control solves a LIBERO task (re-validate on π0.5).
- **M2 — SKIPPED (cost).** Local monolithic baseline campaign is not re-run.
  M4/M5 compare against official OpenPI published LIBERO numbers instead
  (see Evaluation Protocol).
- **M3.** Longor solves `libero_10` task 0 end to end (on π0.5).
- **M4.** Longor beats the published OpenPI π0.5 baseline on `libero_10` task 0
  under matched conditions.
- **M5.** Full LIBERO-10 campaign vs published OpenPI π0.5 suite rates.

## Architecture

Source layout is flat modules under `src/`. Dependencies flow downward only:

`config -> libero_env / vla_client -> plan -> skills -> executor -> planner -> runner`

Module responsibilities:

- `config.py`: single source of runtime defaults and `.env` switches.
- `libero_env.py`: LIBERO wrapper for `observe`, `step`, `success`, and
  structured subtask checks (`check_subtask(step)`).
- `vla_client.py`: HTTP client for the VLA policy service.
- `plan.py`: thin plan helpers (step-array / STOP parsing).
- `skills.py`: static skill catalog text for planner prompting.
- `executor.py`: linear step executor; owns VLA-to-env control loop.
- `planner.py`: LLM planner that emits step arrays or STOP.
- `runner.py`: composition root; single observe -> plan -> execute pass.

Non-negotiable boundaries:

- VLA-to-environment integration lives in `executor.py` in one visible path.
- No hidden retuning layer between policy output and `env.step()` (no remap,
  rescale, sign flips, success rewriting).
- `config.py` is the only place runtime defaults are defined.
- `Executor` does **not** decide global task success and does **not** evaluate
  planner STOP payloads.
- `Runner` handles planner STOP and computes final run verdict from `env.success()`.

## Plan and Skill Contract

### Planner output

Planner output must be one of:

1. Executable steps (NL object/target names from the scene snapshot and goal):
   ```json
   [
     {"skill":"grab","object_name":"alphabet soup"},
     {"skill":"place","object_name":"alphabet soup","target_name":"basket","position":"inside"},
     {"skill":"close","target_name":"microwave"},
     {"skill":"turn_on","target_name":"flat stove"}
   ]
   ```
2. STOP sentinel:
   ```json
   {"stop":"<reason>"}
   ```

`transfer` is not part of the contract.

### Skill set

Runtime skills are:

- `grab(object_name)`
- `place(object_name, target_name, position)`
- `close(target_name)`
- `turn_on(target_name)`

`place.position` values: `inside`, `on`, `center`, `left`, `right`, `top`, `bottom`.

## Execution Contract

- Executor runs steps in order.
- Executor budgeting uses only `MAX_SKILL_STEPS` per step.
- Executor does not enforce episode-level step budget.
- A step ends when `LiberoEnv.check_subtask(step)` is true, or when the
  per-step cap is reached.
- Executor returns one of:
  - `plan_done`
  - `subtask_timeout`
  - `execution_error`

## Subtask Termination Contract

`LiberoEnv.check_subtask(step)` evaluates structured checks (no free-form NLP parsing).
It resolves planner NL names to LIBERO GT object/region IDs before predicate /
grasp checks; exact GT-IDs are also accepted.

- `grab` -> `is_grasping(object_name)`
- `place`:
  - `inside` -> `eval_predicate("in", object_name, target_name)`
  - `on/center/left/right/top/bottom` -> `eval_predicate("on", object_name, target_name)`
- `close` -> `eval_predicate("close", target_name)`
- `turn_on` -> `eval_predicate("turnon", target_name)`

This subtask oracle is deliberate research scaffolding and must be reported as
"given a subtask-completion oracle" in evaluation claims.

## Evaluation Protocol

- Episodes vary LIBERO initial state (`episode i -> init_state[i % count]`).
- Comparison arms must match on task suite/index, initial states, and
  `VLA_ACTIONS_PER_CHUNK` (service-side; restart the VLA service after changing it).
- Campaign-level step ceilings (if used) belong to validation harnesses, not
  the executor runtime loop.
- **Published baseline (replaces local M2):** Official OpenPI π0.5 @ 30k
  finetuned on LIBERO (`gs://openpi-assets/checkpoints/pi05_libero/`, config
  `pi05_libero`), as reported in
  https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md:
  Spatial 98.8%, Object 98.2%, Goal 98.0%, LIBERO-10 92.4%, Average 96.85%.
- Condition matching for M4/M5 claims must follow the OpenPI LIBERO eval loop
  (resolution 256, seed 7, settle/wait steps, suite step caps, 50 trials/task
  for suite campaigns, action chunk / replan horizon aligned with
  `VLA_ACTIONS_PER_CHUNK`). Oracle-based Longor runs must still disclose
  "given a subtask-completion oracle".
- Reference metrics and the machine-checkable checklist live in
  `validation/libero_published_baseline.py`.

## Configuration Contract

`.env` is intentionally minimal: `OPENAI_API_KEY`, `LOG_LEVEL`, `LLM_MODEL`,
`SAVE_VIDEO`.
All other tunables live in `src/config.py`.
Adding a new env var requires explicit SRD-level approval.

## VLA Service Contract

One HTTP policy transport, two backends. Launch via
`python services/run_vla_server.py` (optional `--policy pi05|smolvla`).
Default backend is `pi05` (`config.VLA_POLICY`). Fallback is `smolvla`.

Backend modules:

- `services/pi05_service.py` (default; OpenPI π0.5, `pi05_libero`)
- `services/smolvla_service.py` (fallback; LeRobot SmolVLA)

Both expose the same wire contract:

- `GET /health` -> `{"status":"ok","model":"<id>"}` where `<id>` is `pi05` or
  `smolvla`
- `POST /predict` (multipart):
  - `image` (JPEG), `image_wrist` (JPEG), `task` (string),
    `state` (JSON array of 8 floats)
  - response: `{"actions": [[7 floats], ...]}`
  - `actions` length is capped by `config.VLA_ACTIONS_PER_CHUNK` on the
    **service** before the response is returned. Clients (executor, probes)
    must execute the full returned chunk and must not trim it again.

Backend selection is a launcher/config concern, not a client concern:
`src/vla_client.py` stays backend-agnostic.

Contract changes must be recorded here before implementation changes.

## Future Work

- Replace oracle-based subtask checks with perception-based subtask verification.
- Revisit recursive/sub-loop planning only after baseline stability is shown.
- Extend beyond fixed-base arm manipulation when LIBERO-10 scope is stable.
