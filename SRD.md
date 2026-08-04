# Longor SRD

Single source of truth for architecture and runtime contracts.

Install: `pip install -r deps/requirements-libero.txt && pip install -e .` from the
repo root. Source lives in `src/` as flat top-level modules; `pip install -e .`
and `pyproject.toml`'s `pythonpath = src` both make them importable without
`PYTHONPATH` juggling.

## Goal

Enable robots to perform **long-horizon** tasks by separating **planning** (LLM)
from **execution** (interpreter), so the agent reasons over the *structure* of
the work, not just an ordered step list.

1. The **planner** (LLM) emits a **plan**: a structured algorithm with control
   flow (sequencing, conditionals, loops, feedback points) — not a flat list of
   subtasks.
2. The **executor** walks the plan. Perception steps call **skills**; arm
   manipulation steps run the VLA policy loop directly.
3. Planning is snapshot-grounded. The runner captures an RGB snapshot before the
   first plan and before every replan, then calls `planner.plan(snapshot, stop?)`.
   **Feedback points** pause execution and provide the `stop` context for replans.
   The planner keeps a short `progress_history` across calls. It may emit a fresh
   plan or a sentinel STOP plan (return node with literal `"STOP: <reason>"`).
   The runner detects STOP before `validate_plan` / `executor.run`. The executor is
   stateless across `run(plan)` calls: no `reset`, no continuation. `STEP_BUDGET`
   bounds skill calls **within one** `run(plan)` only. Stopping a stuck run is
   the planner's job (via history + snapshot), not a runner replan cap.
4. Plans are **declarative data, not executable Python**: no `exec` of generated
   code, no `eval` of arbitrary expressions.

**Platform.** LIBERO on Linux + NVIDIA + CUDA (Colab T4 is sufficient). This is
the only supported platform. Arm manipulation is **VLA only**, behind a single
VLA HTTP service.

## Architecture

Eight flat modules. Dependencies point strictly downward — no module imports one
listed above it.

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Every tunable constant; loads `.env`. Imports nothing from Longor. |
| `src/libero_env.py` | `LiberoEnv`: builds `OffScreenRenderEnv`, `observe()`, `step()`, `success()`, `object_names()`, `body_position()`, run video. |
| `src/vla_client.py` | `VLAClient`: HTTP to the policy service. Knows nothing about LIBERO. |
| `src/plan.py` | Pydantic plan AST + `validate_plan()` + `is_a_stop_plan()` / `get_stop_reason()`. |
| `src/skills.py` | `Skill` / `SkillParam` / `SkillRegistry` + LIBERO sim-GT perception skills. |
| `src/executor.py` | Stateless `run(plan)`: walks one plan, owns VLA-to-environment execution, returns a typed `ExecutionStop`; no cross-run memory. |
| `src/planner.py` | `LLMPlanner(goal, catalog, scene_objects)`: `plan(snapshot, stop?) -> dict` (plan AST JSON); requires a snapshot on every call; owns `progress_history` across calls; model/retries from `config`. |
| `src/runner.py` | `Runner`: composition root — owns `LiberoEnv`, `SkillRegistry`; loop is `observe → plan(snapshot) → validate → executor.run(plan)`, then `observe + plan(snapshot, stop)` on feedback; checks sentinel STOP before execute. |

Plus `src/utils.py` (`load_env`, `configure_logging`) and
`services/smolvla_service.py` (the policy server, run in a separate venv).

### Rules

- **The VLA-to-environment integration lives in the executor**, in one visible
  method. An `arm.grab` / `arm.place` node is not a registered skill; the
  executor branches on it and runs the policy loop inline. Debugging manipulation
  means reading one function, not chasing six indirections.
- **No hidden tuning layers.** No action rescaling, axis remapping, sign
  flipping, adaptive scaling, proximity heuristics, or success rewriting between
  the policy output and `env.step()`. The policy's action is applied as-is.
- **No sim-GT fallback in a real-perception path.** Any fallback must be
  explicit, opt-in, and recorded here.
- **`config.py` is the only place defaults are defined.** Reading `os.environ`
  anywhere else is a defect.

### Manipulation loop

The reference OpenPI LIBERO evaluation loop is the contract. `LiberoEnv` and the
executor's manipulation method together must stay behaviourally equivalent to it:

- 7-D OSC action `[dx, dy, dz, drx, dry, drz, gripper]`, applied one action per
  `env.step()`, no scaling.
- `env.set_init_state(suite.get_task_init_states(task_idx)[0])` after `reset()` —
  the benchmark's canonical initial state is part of the eval protocol.
- `env.seed(SEED)` before reset; the seed perturbs object poses even with a fixed
  initial state.
- `SETTLE_STEPS` no-op actions (`[0]*6 + [-1]`) after init, because the simulator
  drops objects onto the table.
- Both camera frames rotated 180 degrees (`[::-1, ::-1]`) to match training
  preprocessing.
- 8-D state: `eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2)`.
- Re-query the policy every `REPLAN_STEPS` actions; discard the rest of the chunk.
- Success is `env.check_success()`.

## Configuration

`.env` holds **only** secrets and switches worth flipping between runs:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Secret; read by the LLM SDK, never by our code. |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |
| `LLM_MODEL` | Planner model id (initial and mid-loop `plan()`). |
| `SAVE_VIDEO` | `1` to write the run video to `config.VIDEO_PATH`. |

Everything else is a constant in `src/config.py`: `VLA_PORT` (8765), `VLA_URL`,
`VLA_TIMEOUT`, `RESOLUTION` (256), `SEED`, `SETTLE_STEPS`, `SETTLE_ACTION`,
`MAX_EPISODE_STEPS` (per suite), `VIDEO_PATH`, `VIDEO_FPS`, `REPLAN_STEPS`,
`MAX_SKILL_STEPS`, `ACTION_CLIP`, `STEP_BUDGET`,
`MAX_PLAN_RETRIES`.

Adding a new constant is fine. Adding a new environment variable requires a
mapped acceptance criterion in this document.

Headless rendering additionally needs `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl` exported in the shell (not application config).

## VLA service contract

One policy service process at a time, `services/smolvla_service.py`, on
`VLA_PORT`. The wire format is fixed:

- `GET /health` → `{"status": "ok", "model": "<id>"}`
- `POST /predict`, multipart: `image` (JPEG), `image_wrist` (JPEG), `task`
  (string), `state` (JSON array of 8 floats)
  → `{"actions": [[7 floats], ...]}`

Changing this contract requires updating this section first.

## Delivery status

The planned refactor phases are complete and retired. This document now focuses
on current-state contracts only.

Current runtime shape:

- `planner.plan(snapshot, stop?) -> dict` (validated plan AST JSON)
- STOP is represented by a sentinel return literal: `"STOP: <reason>"`
- `runner` checks `is_a_stop_plan(...)` before `executor.run(plan)` and reads
  the reason via `get_stop_reason(...)`
- `executor.run(plan)` is stateless across runs

Example entrypoint:

```python
from runner import Runner
from utils import configure_logging

configure_logging()
GOAL = "put the alphabet soup can and the cream cheese box in the basket"
Runner(GOAL, task_suite="libero_10", task_idx=0).run()
print("DONE")
```

## Future work

Deliberately out of scope for the current runtime scope:

- Real (RGB-based) perception replacing sim-GT, then deleting sim-GT.
- LIBERO-10 campaign: independent per-task evaluation for `task_idx` 0..9.
- Recursive sub-planning (a composite skill that re-invokes the planner).
- XLeRobot simulation and hardware backends. The previous XLeRobot code is
  deleted in phase 5; these would start fresh against the contracts above.
- Slimming `deps/requirements-libero.txt` from `lerobot[libero]` to a direct
  LIBERO pin.

## Validation tooling (Colab)

Colab notebooks are **launchers only**: collect secrets, sync the repo, install
deps, start the policy service, run exactly one script under `validation/**`,
print its stdout summary. Orchestration and verdict logic live in the script.

Every script under `validation/**` must print a concise pass/fail summary with
key metrics to stdout and exit 0 on pass, non-zero on fail.
