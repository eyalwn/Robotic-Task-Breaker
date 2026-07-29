# Longor SRD

Single source of truth for architecture, contracts, and the current refactor.

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
3. **Feedback points** capture an RGB snapshot, let the LLM evaluate progress
   against the goal, and route the run to continue, stop, or replan. Replanning
   is bounded.
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
| `src/plan.py` | Pydantic plan AST + `validate_plan()`. |
| `src/skills.py` | `Skill` / `SkillParam` / `SkillRegistry` + LIBERO sim-GT perception skills. |
| `src/executor.py` | Walks the plan; owns the VLA-to-environment loop. |
| `src/planner.py` | `LLMPlanner` (emits plans) + `LLMFeedbackEvaluator` (continue/stop/replan). |
| `src/runner.py` | `Runner`: builds env + planner + executor, then alternates plan and execute. |

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
| `LLM_MODEL` | Planner and feedback model id. |
| `SAVE_VIDEO` | `1` to write the run video to `config.VIDEO_PATH`. |

Everything else is a constant in `src/config.py`: `VLA_PORT` (8765), `VLA_URL`,
`VLA_TIMEOUT`, `RESOLUTION` (256), `SEED`, `SETTLE_STEPS`, `SETTLE_ACTION`,
`MAX_EPISODE_STEPS` (per suite), `VIDEO_PATH`, `VIDEO_FPS`, `REPLAN_STEPS`,
`MAX_SKILL_STEPS`, `ACTION_CLIP`, `MAX_REPLANS`, `STEP_BUDGET`,
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

## Refactor phases

Replaces the former M1-M18 milestone list. History: LIBERO integration, the plan
AST and executor, the LLM planner and feedback evaluator, and SmolVLA and Pi0.5
service baselines were all built and validated before this refactor. The
ManiSkill backend, the XLeRobot hardware stack, the Octo and Pi0.5 services, and
the trace writer are being removed.

### Phase 1 — new environment and policy client, proven in isolation (done)

`src/config.py`, `src/libero_env.py`, `src/vla_client.py`,
`validation/libero_probe.py`. No planner, no executor, no skills, no plan AST.
Nothing deleted.

**Acceptance:** `python validation/libero_probe.py` prints a pass/fail summary
and exits 0, having solved `libero_object` task 0. **Met** on Colab T4 with
SmolVLA (`HuggingFaceVLA/smolvla_libero`):

```
task              'pick up the alphabet soup and place it in the basket'
steps             170 (160 policy actions)
eef displacement  0.3025 m
gripper qpos      0.0323 .. 0.0399
mean |action|     [0.156, 0.361, 0.314, 0.008, 0.031, 0.021, 0.998]
closest approach  alphabet_soup_1: 0.0074 m
closest approach  basket_1: 0.1596 m
PASS: task solved
```

This was also the diagnostic gate for the long-standing VLA/environment defect.
The result settles it: the defect was in the layers this refactor deletes, not in
the service or the checkpoint.

**Baseline to preserve.** This run is the reference any later regression is
measured against. Two properties matter:

- The policy solved a *complete* pick-and-place from a single task string, in one
  uninterrupted rollout. Decomposing the same work into separate `arm.grab` and
  `arm.place` calls gives the policy less context and resets its action chunk
  mid-task, so a decomposed run doing *worse* than this probe is expected and is
  not evidence of a wiring bug. Re-run the probe before blaming the executor.
- 160 actions sufficed for both phases of the task, well inside
  `MAX_EPISODE_STEPS["libero_object"]` (280) and `MAX_SKILL_STEPS` (300).

### Phase 2 — documentation anchored (done)

This document, `README.md`, `.env.example`,
`.cursor/rules/longor-architecture.mdc`.

### Phase 3 — orchestration ported (done)

`src/plan.py`, `src/skills.py`, `src/executor.py`, `src/planner.py`,
`src/runner.py`, registered in `py-modules`. Old tree still present and
untouched. New modules import nothing from `core` / `sim` / `vla` / `robots`.

- `plan.py`: plan AST + `validate_plan()`.
- `skills.py`: registry + sim-GT perception skills on `LiberoEnv`. `arm.grab` /
  `arm.place` are not registered; their catalog lines are appended for the LLM.
- `executor.py`: plan walk, restricted expr eval, `_manipulate()` matching
  `validation/libero_probe.py`. Task strings: strip trailing index, underscores
  → spaces, then `pick up the {name}` / `put the {name} in the {location}`.
  Returns `{"status": "success"}` or `{"status": "failed", "steps": N}`.
  `FeedbackDecision` / `FeedbackOutcome` live here (executor is below planner in
  the dependency order) and are re-exported from `planner.py`.
- `planner.py`: `LLMPlanner` + `LLMFeedbackEvaluator`.
- `runner.py`: `__init__` builds env, VLA client, planner, feedback evaluator,
  registry, and executor; `run()` plans and replans up to `MAX_REPLANS`.

**Acceptance:** `Runner("...", task_suite="libero_10", task_idx=0)` constructs
and `run()` returns a result; no module under `src/core`, `src/sim`, `src/vla`, or
`src/robots` is imported by any new module. **Met** structurally (imports clean,
`Runner.__init__` builds the stack). Live `run()` on `libero_10` task 0 is a
Linux/Colab gate (LIBERO); unit suite stays green against the old tree
(338 passed, 9 skipped).

### Phase 4 — example rewritten

`examples/long_horizon_llm_libero.py` becomes:

```python
from runner import Runner
from utils import configure_logging

configure_logging()
GOAL = "put the alphabet soup can and the cream cheese box in the basket"
result = Runner(GOAL, task_suite="libero_10", task_idx=0).run()
raise SystemExit(0 if result.success else 1)
```

**Acceptance:** the example runs end to end on `libero_10` task 0 and reports a
verdict.

### Phase 5 — old tree deleted

Delete `src/core/**`, `src/sim/**`, `src/vla/**`, `src/robots/**`,
`examples/0_xlerobot_*.py` through `examples/3_xlerobot_*.py`,
`examples/maniskill_pickcube.py`, `validation/m7_wiring_probe.py`,
`validation/m7_direct_task_probe.py`, `validation/m8_policy_health_probe.py`,
`services/octo_service.py`, `services/pi05_service.py`,
`services/run_vla_server.py`, `services/tests/**`, `temp.py`,
`deps/requirements-maniskill.txt`, `deps/requirements-vla-octo-small-1.5-*.txt`,
`deps/requirements-vla-pi05-linux.txt`, `src/longor.egg-info/**`, and the
legacy logger names in `src/utils.py`. Trim `src/utils.py` to `load_env` and
`configure_logging`. Remove `[tool.setuptools.packages.find]` from
`pyproject.toml`.

Also delete every notebook under `colab/` — each launches a deleted script — and
replace them with one launcher for `validation/libero_probe.py`, per
`.cursor/rules/colab-validation.mdc`.

**Acceptance:** the example and the probe still run; nothing imports a deleted
module.

### Phase 6 — tests rewritten

Delete all of `tests/**` and write focused replacements:

| Test | Covers |
| --- | --- |
| `tests/test_plan.py` | AST validation, arg coercion, rejection of malformed plans |
| `tests/test_executor.py` | Control flow (seq/if/while/return/feedback) against a fake env and fake policy |
| `tests/test_executor_expr.py` | Restricted expression evaluator, including error cases |
| `tests/test_vla_client.py` | Wire contract: request shape, response parsing, rejection of bad actions |
| `tests/test_planner.py` | JSON extraction from LLM output, retry on validation failure |

Unit tests stay hermetic: no live sim, no network. Shared fixtures in
`tests/conftest.py`. `pytest -m integration` remains reserved for tests needing a
live service.

**Acceptance:** `pytest` passes with no live sim or network.

## Future work

Deliberately out of scope until the phases above land:

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
