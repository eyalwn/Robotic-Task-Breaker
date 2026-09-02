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
- **Old M4/M5 (beat published OpenPI on LIBERO-10) — CANCELLED.**
- **M4 (SmolVLA / `select_fruit`) — FAILED (policy).** Official
  `lerobot-eval` of `lerobot/smolvla_vlabench` on VLABench `select_fruit`
  (native instruction, SHA `cf588fe60c0c7282174fe979f5913170cfe69017`):
  **0/8** full-horizon episodes (run OOM-killed at 8/10). Checkpoint cannot
  solve the task; Longor path is not diagnostic.
- **M5 — ACTIVE.** Establish a VLABench `select_fruit` baseline and Longor
  subtask run with the official `pi0-fast-primitive-10task-deltachunk`
  checkpoint. The service must use the checkpoint's documented VLABench
  observation and action representation without client or executor retuning.
  **Baseline protocol (fixed):** task `select_fruit`; native
  `env.task_description` on every `/predict`; raw `VLAClient` loop (no
  `Runner`, planner, or executor); apply every action in each returned chunk
  unchanged; `BASELINE_EPISODES` (= 10) independent episodes; stop each episode
  on `env.success()` or `MAX_EPISODE_STEPS`; report aggregate successes/10 and
  success rate. Entrypoint: `examples/run_vlabench_raw_policy.py`; Colab
  launcher: `colab/run_vlabench_raw_policy.ipynb`.

### Current

`pi0fast` / VLABench is the active low-level policy pair.

## Architecture

Source layout is flat modules under `src/`, with the exception of `src/envs/`.
Dependencies flow downward only:

`config -> vla_client / envs.sim_env -> plan -> skills -> executor -> planner -> runner`

Module responsibilities:

- `config.py`: user-selectable runtime composition and cross-module defaults,
  plus `.env` switches. No task identity here.
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
- User-selectable runtime composition and cross-module defaults live in
  `config.py`. Backend-specific implementation constants stay in their owning
  modules.
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

One HTTP policy transport, with each service module specific to one simulator
and checkpoint. Launch via `python services/run_vla_server.py`. The launcher
selects the module exclusively from `config.VLA_POLICY`; it has no policy CLI
argument.

Backend modules expose the same wire contract:

- `GET /health` -> `{"status":"ok","model":"<id>"}`
- `POST /predict` (multipart):
  - `images`: one or more JPEG parts, field name `images`, order matches
    `observe()` cameras
  - `task` (string)
  - `state` (JSON array of floats; length is the env's)
  - response: `{"actions": [[7 floats], ...]}`
  - **Raw action guarantee:** services and `vla_client` validate shape and
    finiteness only. They must never clip, rescale, or otherwise modify a valid
    7-D vector. SmolVLA outputs (including roll targets near π radians) pass
    through unchanged to `env.step()`.
  - `actions` length is capped by `config.VLA_ACTIONS_PER_CHUNK` on the
    **service** before the response is returned. Clients must execute the full
    returned chunk and must not trim it again.

Each service asserts the camera count and state length its checkpoint needs
(HTTP 400 on mismatch). No silent padding/dropping:

- `smolvla` / `lerobot/smolvla_vlabench`: 3 images, 7-D state. Observation keys
  `observation.images.camera1|camera2|camera3` (front, second, wrist).
- `pi05` / `pi05_libero`: 2 images, 8-D state.
- `pi0fast` / `VLABench/pi0-fast-primitive-10task-deltachunk`: VLABench-only.
  Its service must assert the upstream checkpoint's documented camera/state
  schema. OpenPI's native inference output transform converts its generated
  delta chunk to absolute EEF actions before `infer()` returns. Longor sends
  those resulting actions unchanged; neither the service, `executor.py`, nor
  `src/envs/vlabench_env.py` may alter valid actions.

> **Verified 2026-08-28:** `lerobot/smolvla_vlabench` checkpoint metadata
> declares a 6-D state, but its loaded normalizer has 7-D state statistics and
> fails on a 6-D input. This is upstream metadata inconsistency, not an
> environment/service dimension mismatch. Preserve the documented 7-D VLABench
> state; never drop its gripper component based on that metadata.

Backend selection is a `config.py` concern, not a client concern:
`src/vla_client.py` stays backend-agnostic.

Contract changes must be recorded here before implementation changes.
