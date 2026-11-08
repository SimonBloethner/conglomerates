# IOxEE: conglomerates under ergodicity economics

Simulation code for the paper. The model is `collaborative_growth.py`; its flags are documented in
`MODEL_FLAGS.md`, and departures from the work briefs are listed in `diagnostics/deviations.md`.

## Which numbers to use (C37)

| pilot | directory | decision rule | role |
|---|---|---|---|
| **D** | `pilot_d/` | `moment` | **the paper**: base case, 2,055 scenarios (`pilot_design_d.py`; 1,995 in C37, the C38 α = 0.9 cell, the C39 cv block) |
| C | `pilot_c/` | `loggain` | loggain reference (bounded-rationality variant), frozen at C36, 2,145 scenarios (`pilot_design.py`) |

- Every pilot D run of the blocks inherited from pilot C has a loggain twin in pilot C on the same seeds (scenario
  key `twin_id`). `pilot_d/comparison_c_vs_d.csv` holds the paired differences.
- Pilot D adds the `homogeneous` block (every market at IQR 0.2, both rules) and the `nu` block (Student-t, ν ∈
  {2.5, 4}).
- The replay rule is historical: its runs are pilot C's rule-replay block.
- The current handovers are `diagnostics/c37_note.md` (the pilot D results against the loggain twins) and
  `diagnostics/c38_note.md` (C38 diagnostics: event-study matchings, consent by recruit size, the moment rule's
  approximation error, fixed α = 0.9; with `c38_event_study.md`).
  `diagnostics/c39_note.md` adds proposal types and the control-variate rule `decision_rule='cv'`, a variant
  that behaves like the moment rule. Earlier notes:
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
  - `analytics/pilot_d.py --variance | --comparison | --tables`;
  - C38: `analysis/run_traces.py <trace_dir> --matching | --merge-matching --pilot d`,
    `analysis/approximation_error.py <trace_dir>`, `analytics/acceptance_c38.py`, `analytics/fixed_alpha_c38.py`,
    `analytics/identity_c38.py`.
  - C39: `analytics/acceptance_c39.py`, `analytics/cv_c39.py --append-tidy | --table | --acceptance`,
    `analysis/approximation_error.py <trace_dir> --rule cv`, `analytics/identity_c39.py`.
  - C40: `run_burnin_c40.py <i> | --report` (burn-in under the moment rule, `submit_burnin_c40.sh`).
- Without `--pilot`, every script reads and writes pilot C as before.
- Result pickles and traces are not in git. They are kept on Festus under `/groups/m-larch/bt307958/IOxEE_c37`
  (C38: `IOxEE_c38`, which adds `pilot_d/results_c38/` and the identity reruns `pilot_d/results_c38_identity/`);
  C39: `IOxEE_c39`, with `pilot_d/results_c39/` (15 Task 1 reruns, 45 cv runs) and `pilot_d/results_c39_identity/`.
  C40: `IOxEE_c40/diagnostics/burn_in_c40_runs/` (the 20 burn-in runs to T = 16,000, `run_burnin_c40.py`).
