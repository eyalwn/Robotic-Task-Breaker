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

- **M1 — DONE.** Bare π0-FAST solves VLABench `select_fruit` at least once.
- **M2 — DONE.** Longor solves one fixed, held-out Track-1 `select_fruit`
episode.
- **M3 — PLANNED.** Establish bare-VLA and Longor success rates on a harder,
decomposable VLABench task suite. Select the suite before implementation.



### Notes about success-rate evaluation

M1 baseline protocol: `select_fruit`; the first `BASELINE_EPISODES` (= 10)
configs from the installed VLABench package's
`configs/evaluation/tracks/track_1_in_distribution.json`; deterministic
`episode_config` entries (`random_init=False`, `run_mode=eval`); native
`env.task_description` on every `/predict`; raw `VLAClient` loop (no `Runner`,
planner, or executor); full returned chunks with `VLA_ACTIONS_PER_CHUNK = 5`;
stop on `env.success()` or `MAX_EPISODE_STEPS`; report aggregate success rate.
Working verdict is `success_rate > 0.30` (at least 4/10). This is a
cost-bounded subset of the upstream 50-episode protocol.

### Historical findings

- Bare `VLABench/pi0-fast-ft-primitive-10task-deltachunk` achieved **6/10** on
the first ten held-out `select_fruit` Track-1 episodes, using native
instructions and VLABench SHA `cf588fe60c0c7282174fe979f5913170cfe69017`.
- `lerobot/smolvla_vlabench` achieved **0/8** full-horizon `select_fruit`
episodes before the campaign was OOM-killed; it cannot support a diagnostic
Longor comparison.
- In the previous LIBERO scene, `pi05_libero` mostly ignored subtask language:
when asked to grab tomato sauce first, it still approached alphabet soup.



### Current

`pi0fast` / VLABench is the active low-level policy pair.

## Architecture

Source layout is flat modules under `src/`, with the exception of `src/envs/`.
Dependencies flow downward only:

`config -> vla_client / envs.sim_env -> plan -> skills -> executor -> planner -> runner`

Module responsibilities:

- `config.py`: user-selectable runtime composition and cross-module defaults,
plus `.env` switches. No task identity here.
- `.env`: where user-defined runtime switches and secrets live. `src/config.py`
is the only runtime reader of environment variables. `.env` must match
`.env.example`. Adding or changing an env var requires explicit SRD-level
approval.
- `envs/sim_env.py`: `SimEnv` factory. `__new__(task: str, task_idx: int = 0)`
lazy-imports the configured VLABench backend and returns that instance (no
method forwarding). `task_idx` selects the deterministic held-out Track-1
episode configuration for VLABench and defaults to the first config.
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
     {"skill":"transfer","object_name":"banana","target_name":"plate_seen"}
   ]
  ```
2. STOP sentinel:
  ```json
   {"stop":"<reason>"}
  ```



### Skill set

Runtime skills are:

- `transfer(object_name, target_name)`

For π0-FAST/VLABench, `transfer` maps exactly to
`Put the <object_name> into the <target_name>`. Do not add skills until a task
or selected M3 suite requires them.

M2 uses `Runner("select_fruit", task_idx)`, where `task_idx` selects one
held-out `select_fruit` configuration from Track 1.

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
- `pi0fast` / `VLABench/pi0-fast-primitive-10task-deltachunk`: VLABench-only.
Its service must assert the upstream checkpoint's documented camera/state
schema. OpenPI's native inference output transform converts its generated
delta chunk to absolute EEF actions before `infer()` returns. Longor sends
those 7-D actions unchanged through the service/client/executor path.
`src/envs/vlabench_env.py` applies only simulator-native conversion when
stepping the env: camera remap to the policy's documented 3-view order and
gripper thresholding (`>= 0.1` open, else closed) to Franka finger controls.

> **Verified 2026-08-28:** `lerobot/smolvla_vlabench` checkpoint metadata
> declares a 6-D state, but its loaded normalizer has 7-D state statistics and
> fails on a 6-D input. This is upstream metadata inconsistency, not an
> environment/service dimension mismatch. Preserve the documented 7-D VLABench
> state; never drop its gripper component based on that metadata.

Backend selection is a `config.py` concern, not a client concern:
`src/vla_client.py` stays backend-agnostic.

Contract changes must be recorded here before implementation changes.