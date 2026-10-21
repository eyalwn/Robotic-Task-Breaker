# Longor

Longor is a thin orchestration layer that lets an LLM break a long-horizon
robot task into short primitives a VLA can execute.

Project contracts (runtime boundaries and milestones) live in `SRD.md`.

## Run One Demo Episode (Colab)

Use `colab/run_longor_task.ipynb`.

This notebook installs the RoboCasa + GR00T stack, starts:
- upstream GR00T policy server on `:5555`
- Longor HTTP adapter on `:8765`

Then it runs one episode via `examples/run_longor_task.py`.

## Run Longor vs Bare Comparison (Colab)

Use `colab/run_m4.ipynb`.

Workflow:
- copy/setup sections are the same as `run_longor_task`
- set `X` in section 4 (first pass is `X=20`)
- run Longor for `X` episodes
- run bare policy for `X` episodes
- run stats section:
  - early claim if one-sided Fisher exact `p <= 0.01`
  - otherwise continue to `50` episodes per arm and decide at `p <= 0.04`

Artifacts are written under the run artifact directory on Drive.

## Local Development (Windows OK, No Sim)

```bash
python -m venv .venv
source .venv/Scripts/activate        # Linux/macOS: .venv/bin/activate
pip install -r deps/requirements.txt
pip install -e .
pytest
```

Tests are hermetic (no simulator, no network).

## Configuration

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (never commit secrets).
Runtime defaults live in `src/config.py`.
