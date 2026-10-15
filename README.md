# Longor

Longor is a software layer that lets an LLM break a long-horizon robot task into
short steps a VLA can execute.

Note: Project contracts, milestones, and hypothesis live in `SRD.md`.

## Run (Colab, Linux + NVIDIA GPU)

Simulation runs on Colab (L4 verified).

- `colab/run_longor_task.ipynb` — installs the Isaac-GR00T RoboCasa stack,
  starts the GR00T server (`:5555`) and the Longor adapter (`:8765`), runs one
  `open_drawer` episode (`examples/run_longor_task.py`), and writes `run.mp4` and
  logs to Drive. Colab Secrets: `GITHUB_TOKEN`, `OPENAI_API_KEY` (optional
  `HF_TOKEN`).
- `colab/run_robocasa_primitive_gate.ipynb` — M1 upstream-only gate launcher
  (no Longor planner/executor); orchestration and verdict logic live in
  `validation/run_robocasa_primitive_gate.py`.
- `validation/run_drawer_chain_gate.py` — M3 in-scene drawer-chain gate
  launcher (fixed `open` -> `close`, no LLM). It emits per-episode
  `open_ok`/`close_ok`/`success` lines and a final summary to
  `drawer_chain_gate.log`; with `SAVE_VIDEO=1`, it writes per-episode videos as
  `run_epNNN.mp4` under `VIDEO_PATH`'s parent directory.
- `validation/run_m4_pilot.py` — M4 pilot launcher (`bare` and `longor` arms
  on `open_close_drawer`). It writes per-episode records to
  `m4_pilot_episodes.jsonl`, arm summaries to `m4_pilot_summary.json`, and
  per-episode videos as `videos/{arm}_epNNN.mp4` when `SAVE_VIDEO=1`.

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
