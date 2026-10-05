# Longor

Longor is a thin software layer that lets an LLM break a long-horizon robot
task into short steps a VLA can execute.

It is platform-generic: switch LLM, VLA, or simulator with constants in
`src/config.py`. One of each per process. Contracts: `SRD.md`.

## Run (Colab, Linux + NVIDIA GPU)

Simulation runs on Colab (L4 verified).

- `colab/run_longor_task.ipynb` — installs the Isaac-GR00T RoboCasa stack,
  starts the GR00T server (`:5555`) and the Longor adapter (`:8765`), runs one
  `open_drawer` episode (`examples/run_longor_task.py`), and writes `run.mp4` and
  logs to Drive. Colab Secrets: `GITHUB_TOKEN`, `OPENAI_API_KEY` (optional
  `HF_TOKEN`).
- `colab/run_robocasa_primitive_gate.ipynb` — M1 upstream-only gate (no Longor
  planner/executor).

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
switches (`SAVE_VIDEO=1` writes `run.mp4`, `LOG_LEVEL`, `LLM_MODEL`, ...) are
documented in `.env.example`; every default lives in `src/config.py`.

Headless rendering needs `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.
