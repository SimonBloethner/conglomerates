# IOxEE: conglomerates under ergodicity economics

Simulation code for the paper. The model is `collaborative_growth.py`; its flags are documented in
`MODEL_FLAGS.md`, and departures from the work briefs are listed in `diagnostics/deviations.md`.

## Which numbers to use (C37)

| pilot | directory | decision rule | role |
|---|---|---|---|
| **D** | `pilot_d/` | `moment` | **the paper**: base case, 1,995 scenarios (`pilot_design_d.py`) |
| C | `pilot_c/` | `loggain` | loggain reference (bounded-rationality variant), frozen at C36, 2,145 scenarios (`pilot_design.py`) |

- Every pilot D run of the blocks inherited from pilot C has a loggain twin in pilot C on the same seeds (scenario
  key `twin_id`). `pilot_d/comparison_c_vs_d.csv` holds the paired differences.
- Pilot D adds the `homogeneous` block (every market at IQR 0.2, both rules) and the `nu` block (Student-t, ν ∈
  {2.5, 4}).
- The replay rule is historical: its runs are pilot C's rule-replay block.
- The current handover is `diagnostics/c37_note.md`, the pilot D results against the loggain twins. Earlier notes:
  `diagnostics/c34_note.md`–`c36_note.md`.

## Running

- All computation runs on the Festus cluster through SLURM.
- One scenario: `python run_pilot_c.py <id> --pilot d`. Batches: `submit_pilot_d.sh` (list-driven arrays), and
  `submit_traces_d.sh` for the 27 traced runs.
- Outputs:
  - `python summarize_pilot_c.py --pilot d` (tidy, medians);
  - `create_alpha_scatter.py --pilot d`;
  - `analytics/acceptance_c35.py --pilot d`;
  - `analysis/run_traces.py | run_panel.py | run_decomposition.py | burstiness.py <trace_dir> --pilot d`;
  - `analytics/pilot_d.py --variance | --comparison | --tables`.
- Without `--pilot`, every script reads and writes pilot C as before.
- Result pickles and traces are not in git. They are kept on Festus under `/groups/m-larch/bt307958/IOxEE_c37`.
