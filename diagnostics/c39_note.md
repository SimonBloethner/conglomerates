# C39 note: proposal-type counter and the control-variate decision rule

Branch `referee-fixes`, from `f6726e0` (C38 tip plus paper figures, RESULTS_PAGE v2 and bib entries). Commits:
- **C39a**: proposal-type and volatility-bin counters, the 15 Task 1 reruns, Section 1;
- **C39b**: `decision_rule='cv'`, its tests, the 45 cv runs, Section 2.

**What changes in the model.** Existing scenarios keep their rule and their results (Section 0).
- Task 1 adds counters only.
- Task 2 adds a fourth decision rule.

**Where things are.**
- Departures from the brief: `diagnostics/deviations.md` (C39).
- Runs: Festus, `/scratch/bt307958/c39/IOxEE`, copied to `/groups/m-larch/bt307958/IOxEE_c39`.
- New results: `pilot_d/results_c39/` (not in git).
- Pilot C and the earlier pilot D outputs are not regenerated.

## 0. Checks

**Identity.** All 2,010 existing pilot D scenarios (C37 and C38) were rerun on the C39 code
(`pilot_d/results_c39_identity/`, on Festus only) and compared with their pickles (`analytics/identity_c39.py`,
`diagnostics/identity_c39.csv`).
- **Identical.** 2,010 of 2,010 are byte-identical in every field of the reference (runtime fields left out),
  and their tidy rows are identical. The new keys are the only extra fields.
- **Counters consistent in all 2,010.** The type arrays summed over type equal the C38 arrays: integer counts
  exactly, binding-gain sums to rounding. The volatility-bin counts equal the unit+unit counts.
- **One rerun for both tasks.** It was made once, on the code that contains the cv rule; no existing scenario
  reaches that rule. The median runtime ratio is 0.99.
- **Task 1 reruns.** The 15 reruns in `pilot_d/results_c39/` are identical to their identity reruns, field by
  field.

**Pilot C** is unchanged (`md5sum -c` against the C37 checksums).

**Reruns.** The 15 Task 1 reruns reproduce C38: summed over proposal type, their counts equal the C38
`acceptance_by_K_w.csv` row for row (checked in `tests/test_c39_outputs.py`, added with C39b).

**Tests.** The C39a tests (`tests/test_c39_counters.py`, the updated C38 kernel helper) pass on Festus; the full
suite is run at C39b.

## 1. Proposal type beside (K, w) (Task 1)

**Design.**
- Runs: the 15 C38 Task 2 scenarios (main, power law, α = 0.3, three families, 5 reps, moment rule), rerun into
  `pilot_d/results_c39/`.
- Each post-burn-in proposal is also classified by type: solo+solo, solo+unit or unit+unit.
- For unit+unit proposals, the volatility bin records whether the entrant unit (the smaller by capital) has a lower
  mean member IQR than the incumbent unit ("entrant calmer") or not.
- The counters also record which unit holds the **binding member**: the member with the smallest margin, i.e.
  the one whose consent fails first.
- Outputs:
  - `pilot_d/acceptance_by_K_w_type.csv`, `pilot_d/unit_unit_by_volbin.csv`: per run;
  - `*_pooled.csv`: reps pooled;
  - `pilot_d/acceptance_by_K_w_type_report.md`: every K.

### Entrants with w ≥ 0.2: which type

5 reps pooled, α = 0.3. At K ≥ 3 there are no solo+solo proposals.

| family | K | proposals | unit+unit share | acceptance solo+unit / unit+unit | binding mean solo+unit / unit+unit |
|---|---|---|---|---|---|
| normal | 4 | 8,025 | 0.44 | 0.686 / 0.820 | 2.2e-4 / 4.4e-4 |
| normal | 6 | 8,102 | 0.73 | 0.426 / 0.447 | −8.9e-5 / −1.2e-4 |
| normal | 8 | 12,716 | 0.91 | 0.247 / 0.187 | −3.2e-4 / −4.1e-4 |
| normal | 10 | 23,835 | 0.98 | 0.130 / 0.061 | −5.6e-4 / −6.0e-4 |
| normal | 12 | 34,455 | 1.00 | 0.128 / 0.014 | −6.6e-4 / −7.2e-4 |
| laplace | 4 | 12,204 | 0.42 | 0.690 / 0.778 | 4.7e-4 / 6.9e-4 |
| laplace | 6 | 10,198 | 0.72 | 0.486 / 0.431 | −5.9e-5 / −3.1e-4 |
| laplace | 8 | 11,797 | 0.88 | 0.364 / 0.233 | −2.8e-4 / −7.5e-4 |
| laplace | 10 | 15,912 | 0.96 | 0.323 / 0.100 | −3.6e-4 / −1.1e-3 |
| laplace | 12 | 19,906 | 0.99 | 0.196 / 0.041 | −5.9e-4 / −1.3e-3 |
| t₃ | 4 | 18,013 | 0.46 | 0.625 / 0.590 | 9.5e-4 / −5.0e-3 |
| t₃ | 6 | 14,255 | 0.76 | 0.474 / 0.322 | −8.9e-6 / −4.0e-3 |
| t₃ | 8 | 15,407 | 0.89 | 0.419 / 0.167 | −3.8e-4 / −4.3e-3 |
| t₃ | 10 | 17,291 | 0.96 | 0.408 / 0.081 | −7.8e-4 / −4.5e-3 |
| t₃ | 12 | 17,768 | 0.99 | 0.393 / 0.036 | −7.3e-4 / −4.0e-3 |

### Unit+unit proposals by volatility bin

The table shows acceptance (calmer / more volatile entrant) and the share of proposals whose binding member is in
the calmer unit (entrant calmer / entrant more volatile). 5 reps pooled.

| K | normal acceptance | normal: calmer unit binds | laplace acceptance | laplace: calmer unit binds | t₃ acceptance | t₃: calmer unit binds |
|---|---|---|---|---|---|---|
| 4 | 0.786 / 0.804 | 0.21 / 0.33 | 0.752 / 0.763 | 0.22 / 0.34 | 0.549 / 0.558 | 0.25 / 0.40 |
| 5 | 0.466 / 0.663 | 0.14 / 0.51 | 0.469 / 0.637 | 0.16 / 0.49 | 0.358 / 0.460 | 0.22 / 0.47 |
| 6 | 0.253 / 0.506 | 0.13 / 0.53 | 0.309 / 0.517 | 0.14 / 0.53 | 0.237 / 0.355 | 0.21 / 0.49 |
| 7 | 0.139 / 0.364 | 0.14 / 0.52 | 0.204 / 0.402 | 0.16 / 0.52 | 0.169 / 0.284 | 0.23 / 0.50 |
| 8 | 0.081 / 0.258 | 0.17 / 0.48 | 0.145 / 0.318 | 0.17 / 0.50 | 0.118 / 0.210 | 0.25 / 0.48 |
| 10 | 0.029 / 0.105 | 0.25 / 0.36 | 0.063 / 0.176 | 0.22 / 0.43 | 0.062 / 0.121 | 0.29 / 0.45 |
| 12 | 0.008 / 0.028 | 0.33 / 0.25 | 0.030 / 0.082 | 0.29 / 0.35 | 0.030 / 0.064 | 0.33 / 0.40 |

### Is the low acceptance of large entrants a unit–unit phenomenon, and is it the calm side that vetoes?

**Yes, it is a unit–unit phenomenon.**
- Among proposals whose entrant side holds w ≥ 0.2, the unit+unit share grows with K:

  | K | unit+unit share |
  |---|---|
  | 4 | 42–46% |
  | 6 | 72–76% |
  | 8 or more | 88–100% |

- At K ≥ 8 unit+unit proposals are accepted far less often than large standalone entrants:

  | family | unit+unit | large standalone entrants |
  |---|---|---|
  | normal | 0.014–0.187 | 0.13–0.25 |
  | laplace | 0.04–0.23 | 0.20–0.36 |
  | t₃ | 0.04–0.17 | 0.36–0.43 |

- Large standalone entrants are accepted at least as often as floor entrants at the same K (C38: 0.22, 0.31, 0.32
  at K = 8). So the C38 finding about entrants with w ≥ 0.2 is a finding about merging two units.

**No, the calm side does not veto. The volatile side does.**
- **Volatile incumbent.** When the entrant unit is calmer, the incumbent is the volatile, larger unit. The
  binding member is in it in 67–87% of proposals at K = 5–12, and acceptance is about half that of the other bin
  or less (normal, K = 6–10: 0.25 to 0.03, against 0.51 to 0.11).
- **Volatile entrant.** When the entrant unit is the more volatile one, it binds in 47–75% of proposals, although
  it is the smaller unit and usually has fewer members.
- This is the direction of the transfer term in the moment rule's gain (C36 docstring): ½α(Σ_j c_j V(u_j) − V(u_i)).
  - Pooling capital-weighted arithmetic returns moves expected log growth from members with above-average
    variance to those below.
  - So the volatile members' estimated gain is the one that turns negative.
- The low acceptance when the calm unit enters follows from that. The volatile side is then the larger unit, with
  more members whose estimate can fail.
- **Caveat.** The counters do not record the members in each unit, so "fewer members" for the entrant is an
  inference from its smaller capital.
- **Correction to C38.** It inferred that "the transfer term between units of unequal volatility dominates". That
  part stands. Its implied reading that the calm side refuses does not: the side that loses the transfer, the
  volatile one, refuses.
