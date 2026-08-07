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

## Architecture

Source layout is flat modules under `src/`. Dependencies flow downward only:

`config -> libero_env / vla_client -> plan -> skills -> executor -> planner -> runner`

Module responsibilities:

- `config.py`: single source of runtime defaults and `.env` switches.
- `libero_env.py`: LIBERO wrapper for `observe`, `step`, `success`, and episode
  step accounting (`steps` / `max_steps`).
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
- No sim ground-truth object names or poses in the planner/runtime skill path.
  Object naming comes from the RGB snapshot and natural-language goal.

## Episode budget and arm-skill contract

- Episode hard cap is suite `MAX_EPISODE_STEPS` via `LiberoEnv.max_steps`
  (e.g. `libero_10` = 520). Budgeting counts real `env.step()` calls.
- Per-skill cap is `MAX_SKILL_STEPS`, also clamped by remaining episode budget.
- Executor-handled arm skills are attempt primitives, not subtask oracles. Normal
  completion returns:
  `{"status": "attempted", "env_steps": <int>, "task_success": <bool>}`
  where `task_success` is only the global LIBERO BDDL predicate. Execution
  exceptions surface as `StopReason.EXECUTION_ERROR`, not as a fake skill
  failure status.

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

- Run systematic LIBERO-10 per-task evaluation campaigns under the episode
  budget parity contract above.
- Revisit recursive sub-planning only after baseline stability is demonstrated.
