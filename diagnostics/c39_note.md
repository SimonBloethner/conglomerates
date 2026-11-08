# C39 note: proposal-type counter and the control-variate decision rule

Branch `referee-fixes`, from `f6726e0` (C38 tip plus paper figures, RESULTS_PAGE v2 and bib entries). Commits:
- **C39a** `ffde198`: proposal-type and volatility-bin counters, the 15 Task 1 reruns, Section 1;
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

**Pilot D files.**
- `scenarios.json` gains the 45 cv scenarios (ids 2010–2054). Its first 2,010 entries are byte-identical.
- `tidy.csv` gains their 45 rows. Its first 2,010 rows are byte-identical (`cmp` on the prefix).

**Reruns.** The 15 Task 1 reruns reproduce C38: summed over proposal type, their counts equal the C38
`acceptance_by_K_w.csv` row for row (`tests/test_c39_outputs.py`).

**Tests.** All 300 tests pass on Festus. The first full run had one failure: the identity check compared
float sums split by type with no absolute tolerance, and failed on four runs where a sum near zero differed in the
last digits. The check now allows 1e-15 and passes.

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

## 2. `decision_rule='cv'` (Task 2)

**The rule.**
- `member_gain_cv` averages the exact loggain replay minus its first-order term r_τ α (Σ_j c_jτ u_jτ − u_iτ).
- It is exact in expectation and expands nothing. With the weights fixed over the window it equals the loggain
  estimate.
- Consent, exit review and α renegotiation are as under loggain and moment (`MODEL_FLAGS.md`).

**Tests** (`tests/test_cv_rule.py`):
- a two-member window computed by hand under both sharing rules (to 1e-14);
- the −∞ cases;
- on 200 random windows: Δ̂ᶜᵛ − Δ̂ is exactly minus the subtracted term's sample mean, and with frozen weights
  Δ̂ᶜᵛ = Δ̂ (both to 1e-12);
- on a 200,000-period window: loggain, moment and cv agree within 2% of the gain, and with the weight-averaged
  analytical gain within 3%;
- the model runs end to end under cv.

The brief's exact form of test (b) does not hold. Why is in `deviations.md`, and the gaps are below.

### The statistic: Δ̂ᶜᵛ on the C38 Task 3 draws

Same 2,000 draws per trace, pooled over α and reps: 18,000 per row (`pilot_d/approximation_error_cv.csv`).

| family | K | (Δ̂ᶜᵛ − Δ̃)/Δ̃ median [5%, 95%] | (Δ̂ − Δ̃)/Δ̃ median | sign differs, cv vs moment | sign differs, loggain vs moment | gap 1 | gap 2 |
|---|---|---|---|---|---|---|---|
| normal | 2 | −0.018 [−0.051, +0.022] | −0.227 | 0.02% | 2.27% | 0.157 | 0.005 |
| normal | 3 | −0.018 [−0.046, +0.015] | −0.237 | 0.05% | 1.53% | 0.106 | 0.005 |
| normal | 5 | −0.019 [−0.044, +0.011] | −0.254 | 0.06% | 1.49% | 0.070 | 0.005 |
| normal | 8 | −0.020 [−0.049, +0.010] | −0.282 | 0.07% | 1.68% | 0.057 | 0.005 |
| laplace | 2 | −0.014 [−0.081, +0.070] | −0.191 | 0.03% | 1.52% | 0.182 | 0.008 |
| laplace | 3 | −0.015 [−0.071, +0.060] | −0.205 | 0.02% | 1.03% | 0.126 | 0.009 |
| laplace | 5 | −0.014 [−0.064, +0.054] | −0.226 | 0.04% | 0.85% | 0.091 | 0.010 |
| laplace | 8 | −0.015 [−0.061, +0.049] | −0.253 | 0.01% | 0.77% | 0.077 | 0.011 |
| t₃ | 2 | −0.013 [−0.176, +0.212] | −0.175 | 0.04% | 1.36% | 0.178 | 0.010 |
| t₃ | 3 | −0.016 [−0.211, +0.203] | −0.191 | 0.02% | 0.90% | 0.125 | 0.012 |
| t₃ | 5 | −0.019 [−0.249, +0.211] | −0.219 | 0.05% | 0.68% | 0.093 | 0.015 |
| t₃ | 8 | −0.023 [−0.286, +0.213] | −0.246 | 0.11% | 0.83% | 0.085 | 0.019 |

The two gap columns are median absolute differences as a share of median |Δ̂|:
- **Gap 1:** Δ̂ᶜᵛ − Δ̂ against minus C38's replay weight term.
- **Gap 2:** Δ̃ − Δ̂ᶜᵛ against C38's higher-order term.

**What the table shows.**
- **cv sits close to moment, loggain does not.**

  | family | Δ̂ᶜᵛ below Δ̃ (median) | 90% of draws within | Δ̂ below Δ̃ (median) |
  |---|---|---|---|
  | normal | 1.8–2.0% | −5% to +2% | 23–28% |
  | laplace | 1.4–1.5% | −8% to +7% | 19–25% |
  | t₃ | 1.3–2.3% | −29% to +21% | 17–25% |

- Δ̂ᶜᵛ and Δ̃ disagree in sign in at most 0.11% of windows. Δ̂ and Δ̃ disagree in 0.7–2.3%.
- **The exact decomposition.** Since Δ̂ᶜᵛ − Δ̂ is exactly minus the first-order term, it splits the C38 gap:
  - the replay's first-order term is about 16–26% of Δ̃;
  - the omitted higher-order terms, the part of the gap that is the moment rule's own approximation, are about 2%.
- **C38 had this right.** C38's split, with weights frozen at the window's first period, gave the same answer
  but only approximately: freezing the weights also moved the second-order gain (gap 1, 6–18% of |Δ̂|).

### Behaviour: the cv block against moment and loggain

**Runs.** 45 runs, block `cv` (ids 2010–2054): main power-law cells, α ∈ {0.1, 0.3, 0.5}, 5 reps, on the main
cells' seeds.
- Moment: the pilot D main runs on the same seeds.
- Loggain: their pilot C twins.

**Medians over the 5 reps.** The pooled share is the capital share in units. Source: `pilot_d/cv_comparison.csv`;
the full report is `pilot_d/cv_comparison_report.md`.

| family | α | K cv / moment / loggain | pooled share | Hill | acceptance | voluntary exits / period | floor exits / period |
|---|---|---|---|---|---|---|---|
| normal | 0.1 | 3.67 / 3.68 / 3.04 | 0.847 / 0.850 / 0.561 | 1.102 / 1.113 / 1.087 | 0.314 / 0.313 / 0.325 | 0.07 / 0.06 / 2.99 | 34.4 / 34.1 / 34.8 |
| normal | 0.3 | 5.43 / 5.48 / 4.29 | 0.939 / 0.939 / 0.733 | 1.199 / 1.209 / 1.169 | 0.290 / 0.290 / 0.276 | 0.00 / 0.00 / 1.37 | 25.1 / 25.0 / 26.0 |
| normal | 0.5 | 7.71 / 7.70 / 6.34 | 0.957 / 0.956 / 0.912 | 1.315 / 1.342 / 1.316 | 0.266 / 0.269 / 0.235 | 0.00 / 0.00 / 0.24 | 15.3 / 15.6 / 16.1 |
| laplace | 0.1 | 3.65 / 3.71 / 3.00 | 0.851 / 0.853 / 0.634 | 1.097 / 1.105 / 1.098 | 0.425 / 0.428 / 0.413 | 0.02 / 0.02 / 1.78 | 47.2 / 47.4 / 46.8 |
| laplace | 0.3 | 5.34 / 5.40 / 4.00 | 0.913 / 0.914 / 0.782 | 1.195 / 1.195 / 1.165 | 0.398 / 0.402 / 0.361 | 0.00 / 0.00 / 0.85 | 36.3 / 36.3 / 36.9 |
| laplace | 0.5 | 7.78 / 7.76 / 5.95 | 0.936 / 0.934 / 0.906 | 1.337 / 1.344 / 1.329 | 0.376 / 0.379 / 0.321 | 0.00 / 0.00 / 0.18 | 24.1 / 24.5 / 25.1 |
| t₃ | 0.1 | 3.43 / 3.47 / 2.87 | 0.824 / 0.827 / 0.627 | 1.028 / 1.035 / 1.018 | 0.463 / 0.469 / 0.452 | 0.03 / 0.04 / 1.59 | 52.7 / 53.1 / 52.1 |
| t₃ | 0.3 | 4.82 / 4.84 / 3.67 | 0.899 / 0.896 / 0.773 | 1.116 / 1.116 / 1.096 | 0.432 / 0.435 / 0.403 | 0.00 / 0.00 / 0.93 | 42.5 / 42.5 / 43.2 |
| t₃ | 0.5 | 6.88 / 6.78 / 5.06 | 0.922 / 0.920 / 0.884 | 1.230 / 1.221 / 1.182 | 0.404 / 0.409 / 0.359 | 0.00 / 0.00 / 0.31 | 30.2 / 31.2 / 32.3 |

**Paired differences in K** (median [p25, p75] over the reps, paired on the seed):

| family | α | cv − moment | moment − loggain |
|---|---|---|---|
| normal | 0.1 | −0.02 [−0.02, −0.01] | +0.65 [+0.64, +0.67] |
| normal | 0.3 | −0.07 [−0.07, −0.03] | +1.21 [+1.19, +1.22] |
| normal | 0.5 | +0.00 [−0.01, +0.01] | +1.35 [+1.34, +1.36] |
| laplace | 0.1 | −0.05 [−0.07, −0.04] | +0.71 [+0.70, +0.72] |
| laplace | 0.3 | −0.04 [−0.06, −0.04] | +1.38 [+1.30, +1.41] |
| laplace | 0.5 | −0.01 [−0.02, +0.07] | +1.78 [+1.75, +1.95] |
| t₃ | 0.1 | −0.04 [−0.05, −0.04] | +0.61 [+0.60, +0.61] |
| t₃ | 0.3 | −0.02 [−0.03, −0.00] | +1.16 [+1.15, +1.18] |
| t₃ | 0.5 | +0.11 [+0.03, +0.13] | +1.71 [+1.70, +1.73] |

**Acceptance by K at α = 0.3** (`pilot_d/cv_acceptance_by_K.csv`; every K in `pilot_d/cv_acceptance_report.md`):

| K | normal cv / moment / loggain | laplace cv / moment / loggain | t₃ cv / moment / loggain |
|---|---|---|---|
| 3 | 0.897 / 0.897 / 0.553 | 0.871 / 0.870 / 0.593 | 0.780 / 0.760 / 0.569 |
| 4 | 0.667 / 0.677 / 0.375 | 0.674 / 0.680 / 0.409 | 0.589 / 0.589 / 0.393 |
| 5 | 0.470 / 0.475 / 0.275 | 0.528 / 0.536 / 0.313 | 0.476 / 0.482 / 0.300 |
| 6 | 0.340 / 0.349 / 0.210 | 0.425 / 0.430 / 0.248 | 0.390 / 0.396 / 0.242 |
| 8 | 0.202 / 0.205 / 0.129 | 0.286 / 0.295 / 0.169 | 0.277 / 0.282 / 0.163 |
| 10 | 0.114 / 0.115 / 0.073 | 0.192 / 0.199 / 0.109 | 0.192 / 0.201 / 0.114 |
| 12 | 0.043 / 0.043 / 0.038 | 0.116 / 0.121 / 0.069 | 0.123 / 0.130 / 0.072 |

The binding-gain medians agree the same way: cv −8.5e-6 / moment −7.4e-6 / loggain −9.7e-5 at normal K = 5.

### Does the exact-in-expectation rule reproduce the moment rule's scope?

**Yes, to within a small fraction of the gap it closes.**

| measure | cv − moment | moment − loggain |
|---|---|---|
| K (paired median) | −0.07 to +0.11 members (at most 2%) | +0.61 to +1.78 |
| pooled share | within 0.005 | – |
| Hill | within 0.03 | – |
| acceptance | within 0.006 | – |
| voluntary exits per period | 0.00–0.07 for both | 0.18–2.99 under loggain |
| acceptance by K | within 0.02 at every K ≤ 12 | – |

- **The paired IQR of cv − moment** includes zero in 3 of 9 cells (normal α = 0.5, laplace α = 0.5, and t₃ α = 0.3
  at its edge). In the others it is a small systematic shift, of either sign, 2–7% of the moment − loggain
  difference.

**How large is the approximation's effect on behaviour, as opposed to the statistic?**
- **On the statistic** the moment rule's omitted higher-order terms are 1–2% of the gain in median; in t₃ windows they
  range from −29% to +21% (5–95%).
- **On behaviour** they move K by at most 0.11 members and acceptance by at most 0.02.
- **What drives the moment − loggain difference:**
  - The cv rule has the same expectation as loggain and the same noise structure as moment, and it behaves like
    moment.
  - So the moment rule's scope gain over loggain (C37) comes from removing the first-order term, not from the
    expansion.
  - That term is negative in demeaned windows; it is a finite-window bias of the replay.
- **For the paper.** The C37 "verifiability" component of the consent gap, K_moment − K_loggain, is this bias
  rather than estimation noise. The moment rule and the cv rule are interchangeable for every outcome reported
  here.
