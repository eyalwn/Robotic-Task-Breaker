# M4 parity gate artifacts

Run both Colab notebooks, then compare artifacts with:

```bash
python validation/m4_parity_capture.py verdict \
  --baseline artifacts/m4_parity/lerobot_baseline.json \
  --longor artifacts/m4_parity/longor_subtask.json
```

## Notebooks

| Track | Notebook | Episodes | Prompt |
| ----- | -------- | -------- | ------ |
| Official LeRobot baseline | `colab/run_lerobot_baseline.ipynb` | 10 | native VLABench task string |
| Longor subtask path | `colab/run_vlabench_task.ipynb` | 1 | planner → `pick up <fruit>` |

VLABench SHA (both): `cf588fe60c0c7282174fe979f5913170cfe69017`

Policy (both): `lerobot/smolvla_vlabench`

## Stop conditions

- `official_reference_fails` — baseline success rate is 0; pivot away from this checkpoint.
- `repaired_longor_succeeds` — baseline OK and Longor subtask run succeeded.
- `longor_fails_baseline_succeeds` — run native-prompt follow-up before blaming model quality.

Copy JSON outputs from Colab into this directory before running the verdict command.
