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
- `Runner` closes the loop (`observe -> plan -> execute -> feedback -> replan`)
  until success, failure, or planner STOP.
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
- `libero_env.py`: LIBERO wrapper for `observe`, `step`, `success`, and object
  metadata access.
- `vla_client.py`: HTTP client for the VLA policy service.
- `plan.py`: plan AST models, validation, and STOP helpers.
- `skills.py`: skill interfaces/registry and LIBERO state/perception skills.
- `executor.py`: stateless plan interpreter; owns VLA-to-environment execution.
- `planner.py`: LLM planner that emits validated AST plans from snapshot and
  feedback context.
- `runner.py`: composition root and orchestration loop for one task.

Non-negotiable boundaries:

- VLA-to-environment integration lives in `executor.py` in one visible path.
- No hidden retuning layer between policy output and `env.step()` (no remap,
  rescale, sign flips, or success rewriting).
- `config.py` is the only place runtime defaults are defined.

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

- Replace sim-GT perception skills with real RGB perception and retire sim-GT
  paths.
- Run systematic LIBERO-10 per-task evaluation campaigns.
- Revisit recursive sub-planning only after baseline stability is demonstrated.
