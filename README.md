# Longor

Longor is a software layer that lets an LLM break a long-horizon robot task into
short steps a VLA can execute.

Note: Project contracts, milestones, and hypothesis live in `SRD.md`.

## Run (Colab, Linux + NVIDIA GPU)

Simulation runs on Colab (L4 verified).

- `colab/run_longor_task.ipynb` — minimal single-task launcher: installs the
  Isaac-GR00T RoboCasa stack, starts GR00T (`:5555`) and the Longor adapter
  (`:8765`), then runs one episode via `examples/run_longor_task.py` (default
  task `open_drawer`; notebook uses `open_close_drawer` for M4-path debug).
  Artifacts: `run.mp4`, `run_longor_task.log` on Drive. Colab Secrets:
  `GITHUB_TOKEN`, `OPENAI_API_KEY` (optional `HF_TOKEN`).
- `colab/run_m4_pilot.ipynb` — M4 batch launcher (`validation/run_m4_pilot.py`,
  bare + longor arms). Writes `m4_pilot_episodes.jsonl`, `m4_pilot_summary.json`,
  and optional per-episode videos under one artifact directory; share that path
  for analysis.
- `colab/run_robocasa_primitive_gate.ipynb` — M1 upstream-only gate launcher
  (no Longor planner/executor); orchestration and verdict logic live in
  `validation/run_robocasa_primitive_gate.py`.
- `validation/run_drawer_chain_gate.py` — M3 in-scene drawer-chain gate
  launcher (fixed `open` -> `close`, no LLM). It emits per-episode
  `open_ok`/`close_ok`/`success` lines and a final summary to
  `drawer_chain_gate.log`; with `SAVE_VIDEO=1`, it writes per-episode videos as
  `run_epNNN.mp4` under `VIDEO_PATH`'s parent directory.
- `validation/run_m4_pilot.py` — M4 pilot orchestration (paired `bare` and
  `longor` arms on `open_close_drawer`, one env per pair restored from a
  post-reset checkpoint, horizon-only Longor cap, plan-once). Same artifact
  layout as the M4 Colab notebook.
- `validation/probe_pair_restore.py` — restore-fidelity probe (no VLA/LLM):
  replays one scripted action sequence pristine, after a clean restore, and
  after a dirty restore, then writes `pair_restore_probe.json` with verdict
  `EXACT`, `CLEAN_ONLY`, or `INEXACT`. Optional cell in the M4 notebook.

M4/M5 Longor runtime uses backend `env.horizon` as the only hard step cap;
`MAX_SKILL_STEPS` (720/400) is observational telemetry only. M3 gate scripts
pass explicit `skill_step_limits` to enforce per-skill caps.

## Local development (Windows OK, no sim)

```bash
python -m venv .venv
source .venv/Scripts/activate        # Linux/macOS: .venv/bin/activate
pip install -r deps/requirements.txt
pip install -e .
pytest
```

Tests are hermetic: no sim, no network.

## Configuration

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (never commit it). Other
switches are documented in `.env.example`; runtime defaults live in
`src/config.py`.

Headless rendering needs `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.
