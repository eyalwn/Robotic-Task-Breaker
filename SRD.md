# Longor SRD

Architecture and runtime contracts. Setup and run instructions: `README.md`.

## Goal

Thin LLM planner + VLA executor layer for long-horizon sim tasks. Not an AST
system.

## Milestones

- **M1 — DONE.** Upstream-only RoboCasa / GR00T N1.6 OpenDrawer gate
  (`colab/run_robocasa_primitive_gate.ipynb`): pass = ≥7/10 episodes. Observed
  8/10 (10×720 steps, 8 actions/call). Artifacts on Drive, not in repo.
- **M2 — DONE.** Longor OpenDrawer on RoboCasa with GR00T N1.6-3B through the
  HTTP VLA contract (`colab/run_longor_task.ipynb` → `examples/run_longor_task.py`).
  Succeeded in one `Runner` episode. Report results as "given a
  subtask-completion oracle".
- **M3 — not formalized.**

## Architecture

Flat modules under `src/`, sim backends under `src/envs/` only. Dependency order:

`config → vla_client / envs.sim_env → plan → skills → executor → planner → runner`

- `config.py`: the only place defaults and `.env` switches live.
- `SimEnv(task, task_idx=0)`: lazy-imports the configured backend into
  `self._env` and forwards `observe`, `step`, `success`, `check_subtask`,
  `close`, plus `steps` and `task_description`.
- VLA integration lives in `executor.py` (one path). The action reaches
  `env.step()` as-is: no rescaling or remapping. `Runner` owns STOP and the
  final verdict; `Executor` does not.
- `observe()` → `(images, state)`: tuple of lossless PNG bytes + float list.
  The planner uses `images[0]`.
- Sim ground truth is used only for subtask/episode termination checks
  (`check_subtask`, `success`).
- Unsupported backend API: hard stop with `'not supported for this env'`.
- No `*args`/`**kwargs` in `src/**` or `services/**`.

## Plan and skills

Planner JSON: step array or `{"stop":"<reason>"}`.

Skill: `open(target_name)`; VLA prompt `open the <target_name>`.

## VLA service contract

Launch: `python services/run_vla_server.py` (`config.VLA_POLICY` selects the
service; today only `groot`, which bridges HTTP `:8765` to the upstream GR00T
ZMQ server `:5555`).

- `GET /health` → `{"status":"ok","model":"<id>"}`; optional int `action_dim`.
- `POST /predict` multipart: PNG `images` (order = `observe()`), `task` string,
  `state` JSON float array → `{"actions": [[…], …]}`.
- Services cap chunk length at `config.VLA_ACTIONS_PER_CHUNK` (default 8, as in M1).
- **Raw action guarantee:** validate shape and finiteness only; no rescale/remap.
  `vla_client` infers vector length from the first action and requires every
  vector in the chunk to match.
- Backend-specific camera/state asserts → HTTP 400; no silent padding.

`src/vla_client.py` stays backend-agnostic. Contract changes are recorded here
first.

## Historical findings (legacy experiments)

VLABench + π0-FAST / SmolVLA support was removed after M2; see git history.

- Bare `VLABench/pi0-fast-primitive-10task-deltachunk` achieved **6/10** on
  held-out `select_fruit` Track-1 episodes.
- Longor on VLABench `cook_dishes` (10 fixed episodes) achieved **0/10**;
  failure split: **6/10** physical stalls and **4/10** oracle-grounding mismatch.
- `lerobot/smolvla_vlabench` achieved **0/8** full-horizon `select_fruit`
  before OOM termination.
- Provenance is external Colab artifacts (not auditable from the repo).
