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

- **M1 — DONE.** Closed-loop VLA control solves a LIBERO task.
- **M2 — DONE.** Monolithic baseline success rate on `libero_10` task 0.
- **M3.** Longor solves `libero_10` task 0 end to end.
- **M4.** Longor beats the M2 baseline on `libero_10` task 0.
- **M5.** Full LIBERO-10 campaign.

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
  - `actions` length is capped by `config.VLA_ACTIONS_PER_CHUNK` on the
    **service** before the response is returned. Clients (executor, probes)
    must execute the full returned chunk and must not trim it again.

Contract changes must be recorded here before implementation changes.

## Future Work

- Replace oracle-based subtask checks with perception-based subtask verification.
- Revisit recursive/sub-loop planning only after baseline stability is shown.
- Extend beyond fixed-base arm manipulation when LIBERO-10 scope is stable.
