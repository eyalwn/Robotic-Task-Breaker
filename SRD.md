# Longor SRD

Single source of truth for Longor runtime architecture and contracts.
Operational runbook details live in `README.md`.

## Goal

Longor is a thin software layer that lets an LLM break a long-horizon robot
task into short steps a VLA can execute.

Hypothesis (next milestones): Longor beats a bare VLA on long-horizon tasks.
The hypothesis is platform-agnostic; benches and checkpoints are how we test it.

This is intentionally not an AST system.

## Milestones

### Prior (LIBERO)

- **M1 — DONE.** Closed-loop VLA control solves a LIBERO task.
- **M2 — SKIPPED (cost).** Local monolithic baseline campaign was not re-run.
- **M3 — DONE.** Longor + `pi05_libero` solved `libero_10` task 0 end to end.
- **Finding.** `pi05_libero` mostly ignores subtask language in that scene
  (same-scene object swap: tomato sauce vs alphabet soup). It still goes for
  alphabet soup when asked to grab tomato sauce first.
- **Old M4/M5 (beat published OpenPI on LIBERO-10) — CANCELLED.** migrated to VLAbench + SmolVLA (`lerobot/smolvla_vlabench`).

### Current

- **M4.** SmolVLA solves `VLAbench - select_fruit` via `colab/run_vlabench_task.ipynb`.

Further VLAbench tasks are planned after M4.

## Architecture

Source layout is flat modules under `src/`, with the exception of `src/envs/`.
Dependencies flow downward only:

`config -> vla_client / envs.sim_env -> plan -> skills -> executor -> planner -> runner`

Module responsibilities:

- `config.py`: runtime defaults and `.env` switches. No task identity here.
- `.env`: where user-defined configurations + secrets live. It is intentionally minimal: `OPENAI_API_KEY`, `LOG_LEVEL`, `LLM_MODEL`, `SAVE_VIDEO`. `.env` must match `.env.example`. Adding a new env var requires explicit SRD-level approval.
- `envs/sim_env.py`: `SimEnv` factory. `__new__(task: str, task_idx: int = 0)`
  lazy-imports one backend from `config.SIM_ENV` and returns that instance (no
  method forwarding). `task_idx` defaults to `0` (first task) when omitted.
  VLAbench uses `task` only; LIBERO maps to `LiberoEnv(task, task_idx)`.
- `vla_client.py`: HTTP client for the VLA policy service (backend-agnostic).
- `plan.py`: step-array / STOP parsing.
- `skills.py`: static skill catalog text for planner prompting.
- `executor.py`: linear step executor; owns VLA-to-env control loop.
- `planner.py`: LLM planner that emits step arrays or STOP.
- `runner.py`: composition root; single observe -> plan -> execute pass.
  `Runner(task: str, task_idx: int = 0)` forwards those args to `SimEnv`.

Non-negotiable boundaries:

- VLA-to-environment integration lives in `executor.py` in one visible path.
- No hidden retuning layer between policy output and `env.step()` (no remap,
  rescale, sign flips, success rewriting).
- `config.py` is the only place runtime defaults are defined.
- Runtime signatures in `src/**` and `services/**` name every parameter; no
  `*args` / `**kwargs`.
- `Executor` does **not** decide global task success and does **not** evaluate
  planner STOP payloads.
- `Runner` handles planner STOP and computes final run verdict.

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

Do not add skills until a task (or the user) requires it.

## VLA Service Contract

One HTTP policy transport, two backends. Launch via
`python services/run_vla_server.py` (optional `--policy pi05|smolvla`).

Backend modules expose the same wire contract:

- `GET /health` -> `{"status":"ok","model":"<id>"}` where `<id>` is `pi05` or
  `smolvla`
- `POST /predict` (multipart):
  - `images`: one or more JPEG parts, field name `images`, order matches
    `observe()` cameras
  - `task` (string)
  - `state` (JSON array of floats; length is the env's)
  - response: `{"actions": [[7 floats], ...]}`
  - `actions` length is capped by `config.VLA_ACTIONS_PER_CHUNK` on the
    **service** before the response is returned. Clients must execute the full
    returned chunk and must not trim it again.

Each service asserts the camera count and state length its **default
checkpoint** needs (HTTP 400 on mismatch). No silent padding/dropping:

- `smolvla` / `lerobot/smolvla_vlabench`: 3 images, 7-D state. Observation keys
  `observation.images.camera1|camera2|camera3` (front, second, wrist).
- `pi05` / `pi05_libero`: 2 images, 8-D state.

Backend selection is a launcher/config concern, not a client concern:
`src/vla_client.py` stays backend-agnostic.

Contract changes must be recorded here before implementation changes.
