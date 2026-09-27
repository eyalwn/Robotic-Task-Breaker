# Longor SRD

Architecture and runtime contracts. Operational setup: `README.md`.

## Goal

Thin LLM planner + VLA executor layer for long-horizon sim tasks. Not an AST
system.

## Milestones

- **M1 — DONE.** Upstream-only RoboCasa / GR00T N1.6 primitive suitability
  gate (Colab `colab/run_robocasa_primitive_gate.ipynb`). No Longor
  planner/executor/`VLAClient`. Artifacts on Drive, not in repo.
- **M2 — ACTIVE.** Longor OpenDrawer on RoboCasa (`OpenDrawer_PandaOmron_Env`)
  with GR00T N1.6-3B via existing HTTP VLA contract. Colab launcher:
  `colab/run_longor_task.ipynb` → `examples/run_longor_task.py`.
- **M3 — not formalized.** Longer-horizon RoboCasa or alternate sim targets are
  deferred until M2 gate evidence is stable; no committed M3 contract yet.

**M1 pass:** ≥7/10 successes in upstream `rollout.log` (`GATE_PASS` in
`verdict.txt`). Reference run: checkpoint `nvidia/GR00T-N1.6-3B`, embodiment
`ROBOCASA_PANDA_OMRON`, env `robocasa_panda_omron/OpenDrawer_PandaOmron_Env`,
10×720 steps, 8 actions/call, upstream server `127.0.0.1:5555`. Observed **8/10**
(external verdict path).

**M2 boundary:** Same Longor stack as VLABench (`Runner` → planner → executor
→ `VLAClient` → `SimEnv`). GR00T adapter in `services/groot_service.py` bridges
HTTP `:8765` to upstream ZMQ `:5555`. RoboCasa wire keys/dims live in
`src/envs/robocasa_env.py` (pinned n1d6 PandaOmron processor order). Skill
`open_drawer`; VLA prompt `Open the drawer` until verified against pinned
upstream. Subtask completion uses documented sim drawer oracle (report as
"given a subtask-completion oracle"). No retuning between policy output and
`env.step()`. Validation: `validation/run_robocasa_m2_gate.py` — Stage A raw
OpenDrawer ≥7/10; Stage B one `Runner` episode; live stages need Linux +
`SIM_ENV = "robocasa"` in `src/config.py` and `VLA_POLICY=groot` in `.env`.
Gate artifacts under `validation/artifacts/robocasa_m2_gate/` (gitignored).

VLAbench + π0-FAST remains in tree as alternate stack (`SIM_ENV = "vlabench"`
in `src/config.py`, `VLA_POLICY=pi0fast` in `.env`).

## Historical findings (legacy experiments)

- Bare `VLABench/pi0-fast-primitive-10task-deltachunk` achieved **6/10** on
  held-out `select_fruit` Track-1 episodes.
- Longor on VLABench `cook_dishes` (10 fixed episodes) achieved **0/10**;
  failure split: **6/10** physical stalls and **4/10** oracle-grounding mismatch.
- `lerobot/smolvla_vlabench` achieved **0/8** full-horizon `select_fruit`
  before OOM termination.
- Provenance for these historical results is external Colab artifacts (not
  fully auditable from committed repo files).

## Architecture

Flat modules under `src/`, sim backends under `src/envs/` only. Dependency order:

`config → vla_client / envs.sim_env → plan → skills → executor → planner → runner`

- `config.py`: cross-module defaults and `.env` switches only (no task identity,
  no backend wire schemas).
- `SimEnv.__new__(task, task_idx=0)`: lazy-imports configured backend; no method
  forwarding on the facade.
- VLA integration in `executor.py` (one path). `Runner` owns STOP and final
  verdict; `Executor` does not.
- `observe()` → `(images, state)` with JPEG tuple + float list; planner uses
  `images[0]`.
- Missing backend API: hard stop with `'not supported for this env'`.
- No `*args`/`**kwargs` in `src/**` or `services/**`.

## Plan and skills

Planner JSON: step array or `{"stop":"<reason>"}`.

Skills:

- `open_drawer` — RoboCasa slice (no extra fields).
- `transfer(object_name, target_name)` — VLABench; VLA string
  `Put the <object> into the <target>`.

## VLA service contract

Launch: `python services/run_vla_server.py` (`config.VLA_POLICY` selects module).

- `GET /health` → `{"status":"ok","model":"<id>"}`; optional int `action_dim`.
- `POST /predict` multipart: JPEG `images` (order = `observe()`), `task` string,
  `state` JSON float array → `{"actions": [[…], …]}`.
- Services cap chunk length at `config.VLA_ACTIONS_PER_CHUNK`.
- **Raw action guarantee:** validate shape and finiteness only; no rescale/remap.
  `vla_client` infers vector length from the first action in a chunk and requires
  all vectors in the chunk to match.
- Backend-specific camera/state asserts → HTTP 400; no silent padding.

`src/vla_client.py` stays backend-agnostic. Contract changes recorded here first.

**SmolVLA note:** checkpoint metadata may say 6-D state; loaded normalizer
expects 7-D VLABench state — do not drop gripper dim.
