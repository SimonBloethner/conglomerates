# C38 note: three diagnostics on the frozen model, one optional cell

Branch `referee-fixes`, from C37 (`b4b9ea2`). The model's behaviour is unchanged. Task 2 adds counters to the
merger kernel, and the identity check confirms they change nothing else (Section 0).

Commits:
- **C38a** `281b9c4`: event study (Task 1), note `diagnostics/c38_event_study.md`;
- **C38b** `5e446cc`: (K, w) counters, identity check, the 15 Task 2 runs;
- **C38c**: approximation error (Task 3), the α = 0.9 cell (Task 4), this note.

**Where things are.**
- Departures from the brief are in `diagnostics/deviations.md` (C38).
- Runs were on Festus in `/scratch/bt307958/c38/IOxEE`, copied to `/groups/m-larch/bt307958/IOxEE_c38`.
- Pilot C and the C37 pilot D outputs are not regenerated.

## 0. Checks

**Identity.** All 1,995 pilot D scenarios were rerun on the C38 code (`pilot_d/results_c38_identity/`, on Festus
only) and compared with their C37 pickles (`analytics/identity_c38.py`, `diagnostics/identity_c38.csv`):
- 1,995 of 1,995 are byte-identical as whole objects without the runtime fields and the C38 keys, and their tidy
  rows are identical;
- in all 1,995 the new counters are consistent: summed over w they equal `proposals_by_K` and `mergers_by_K`, and
  the mean w of each column lies in its bin;
- the 15 Task 2 runs in `pilot_d/results_c38/` are identical to their identity reruns, field by field;
- the median runtime ratio C38/C37 is 0.99.

**Pilot D files.**
- `scenarios.json` gains the 15 α = 0.9 scenarios (ids 1995–2009); its first 1,995 entries are byte-identical.
- `tidy.csv` gains their 15 rows; its 1,995 C37 rows are byte-identical (checked with `cmp` on the prefix).

**Pilot C** is unchanged (`md5sum -c` against the C37 checksums).
- The first check failed on ten pilot C figures. The cause was the C38 code sync, which copied the git versions
  over the Festus copies. Those Festus copies had been regenerated there with different bytes, and the C37
  checksums are of them.
- The Festus copies were restored from the C37 tree and the check passes. The figures in git are unchanged.

**Tests.** All 281 tests pass on Festus.

## 1. Event study (Task 1)

The full note is `diagnostics/c38_event_study.md`. In short:

**The sign comes from the pre-period.**
- Under the current matching (raw share at entry), the owner-path DiD is negative because the pre-entry gap
  (joiner − control) is positive and larger than the post-entry gap.
- The controls, not the joiners, cause it. Raw-share matching picks controls whose share was propped up by floor
  top-ups: they were recapitalised 0.55–1.11 pp per period more than the joiners in the pre-window.
- **Correction to C37.** The C37 note blamed the joiners' pre-entry hits. That has the sign the wrong way.

**Every alternative matching makes the owner DiD positive, in all 27 traces.**

| matching | owner DiD medians |
|---|---|
| A, owner-path share | +0.38 to +0.56 pp |
| B, hit-free window | +0.14 to +0.27 pp |
| C, same hit count | +0.31 to +0.44 pp |

**The post-entry gap barely moves across matchings.** It is +0.01 to +0.26 pp and rises with α and tail weight.
- Matchings A and C overshoot: once the controls are like the joiners, the joiners show a dip before entry.
- B is closest to parallel pre-trends.

## 2. Consent profile by recruit size (Task 2)

**Design.**
- Runs: main block, power law, α = 0.3, 5 reps per family, moment rule. They are reruns of pilot D scenarios
  165–169, 345–349 and 525–529 on the C38 code, in `pilot_d/results_c38/`.
- Every post-burn-in proposal is classified by its entrant and the entrant's capital share w of the prospective
  unit (`MODEL_FLAGS.md`, C38).
  - **"Floor":** a standalone entrant whose market share is below twice the floor's minimum share. Why this differs
    from the brief's literal definition is in `deviations.md`.
  - **w ≥ 0.2:** every other proposal whose entrant holds at least a fifth of the capital.
- Outputs:
  - `pilot_d/acceptance_by_K_w.csv`: per run;
  - `pilot_d/acceptance_by_K_w_pooled.csv`: reps pooled;
  - `pilot_d/acceptance_by_K_w_report.md`: the full tables;
  - `diagnostics/pilot_d_acceptance_by_K_w.png`.

**Columns in the tables below.**
- *Incumbent gain:* the exact second-order gain of an incumbent when the entrant has weight w̄ (proportional
  sharing, equal variances).
- *Margin:* that gain minus the marginal cost Φ(K) − Φ(K−1). It is the analytical counterpart of the binding gain
  when the binding member is an incumbent.
- *Equal-weight gain:* the Section 2 marginal gain into K.
- The binding gain is the minimum over members of the estimated proposed gain minus max(0, current).

**normal** (α = 0.3)

| K | floor share | w̄ floor | acc. floor | binding mean, floor | margin, floor | acc. w ≥ 0.2 | binding mean, w ≥ 0.2 | equal-weight gain | incumbent gain, floor | marginal cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.81 | 0.220 | 0.900 | 6.6e-4 | 6.2e-4 | 0.896 | 7.6e-4 | 9.3e-4 | 6.5e-4 | 2.9e-5 |
| 4 | 0.73 | 0.118 | 0.665 | 1.8e-4 | 2.1e-4 | 0.746 | 3.2e-4 | 4.6e-4 | 2.4e-4 | 3.1e-5 |
| 5 | 0.72 | 0.067 | 0.461 | 3.9e-5 | 7.0e-5 | 0.579 | 5.5e-5 | 2.8e-4 | 1.0e-4 | 3.3e-5 |
| 6 | 0.70 | 0.044 | 0.342 | −5.1e-6 | 2.1e-5 | 0.441 | −1.1e-4 | 1.9e-4 | 5.6e-5 | 3.4e-5 |
| 7 | 0.67 | 0.034 | 0.267 | −2.2e-5 | −6.5e-8 | 0.297 | −2.6e-4 | 1.3e-4 | 3.5e-5 | 3.6e-5 |
| 8 | 0.62 | 0.029 | 0.218 | −3.1e-5 | −1.1e-5 | 0.193 | −4.0e-4 | 9.9e-5 | 2.6e-5 | 3.7e-5 |
| 10 | 0.45 | 0.023 | 0.156 | −4.2e-5 | −2.2e-5 | 0.062 | −6.0e-4 | 6.2e-5 | 1.6e-5 | 3.8e-5 |
| 12 | 0.21 | 0.019 | 0.116 | −4.9e-5 | −2.9e-5 | 0.014 | −7.2e-4 | 4.2e-5 | 1.1e-5 | 4.0e-5 |

**laplace** (α = 0.3)

| K | floor share | w̄ floor | acc. floor | binding mean, floor | margin, floor | acc. w ≥ 0.2 | binding mean, w ≥ 0.2 | equal-weight gain | incumbent gain, floor | marginal cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.77 | 0.201 | 0.870 | 1.2e-3 | 1.1e-3 | 0.877 | 1.4e-3 | 1.8e-3 | 1.2e-3 | 2.9e-5 |
| 4 | 0.68 | 0.110 | 0.670 | 3.6e-4 | 4.0e-4 | 0.727 | 5.7e-4 | 8.9e-4 | 4.3e-4 | 3.1e-5 |
| 5 | 0.67 | 0.064 | 0.530 | 1.1e-4 | 1.6e-4 | 0.579 | 1.0e-4 | 5.4e-4 | 1.9e-4 | 3.3e-5 |
| 6 | 0.65 | 0.043 | 0.431 | 2.6e-5 | 7.0e-5 | 0.446 | −2.4e-4 | 3.6e-4 | 1.0e-4 | 3.4e-5 |
| 7 | 0.63 | 0.031 | 0.361 | −9.7e-6 | 2.8e-5 | 0.335 | −4.9e-4 | 2.6e-4 | 6.4e-5 | 3.6e-5 |
| 8 | 0.61 | 0.025 | 0.309 | −2.5e-5 | 7.3e-6 | 0.249 | −7.0e-4 | 1.9e-4 | 4.4e-5 | 3.7e-5 |
| 10 | 0.52 | 0.018 | 0.236 | −4.1e-5 | −1.3e-5 | 0.108 | −1.1e-3 | 1.2e-4 | 2.5e-5 | 3.8e-5 |
| 12 | 0.38 | 0.015 | 0.189 | −5.0e-5 | −2.3e-5 | 0.042 | −1.3e-3 | 8.1e-5 | 1.7e-5 | 4.0e-5 |

**t₃** (α = 0.3)

| K | floor share | w̄ floor | acc. floor | binding mean, floor | margin, floor | acc. w ≥ 0.2 | binding mean, w ≥ 0.2 | equal-weight gain | incumbent gain, floor | marginal cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.77 | 0.194 | 0.761 | 1.2e-3 | 1.2e-3 | 0.782 | 1.1e-3 | 2.0e-3 | 1.3e-3 | 2.9e-5 |
| 4 | 0.68 | 0.104 | 0.593 | 1.2e-3 | 4.3e-4 | 0.609 | −1.8e-3 | 1.0e-3 | 4.6e-4 | 3.1e-5 |
| 5 | 0.65 | 0.063 | 0.495 | 7.6e-4 | 1.8e-4 | 0.488 | −2.5e-3 | 6.0e-4 | 2.1e-4 | 3.3e-5 |
| 6 | 0.63 | 0.043 | 0.416 | 5.7e-4 | 8.2e-5 | 0.359 | −3.0e-3 | 4.0e-4 | 1.2e-4 | 3.4e-5 |
| 7 | 0.61 | 0.032 | 0.363 | 2.8e-4 | 3.7e-5 | 0.279 | −3.5e-3 | 2.9e-4 | 7.2e-5 | 3.6e-5 |
| 8 | 0.58 | 0.025 | 0.321 | 1.8e-4 | 1.2e-5 | 0.193 | −3.9e-3 | 2.2e-4 | 4.9e-5 | 3.7e-5 |
| 10 | 0.49 | 0.018 | 0.262 | 1.6e-4 | −1.1e-5 | 0.094 | −4.3e-3 | 1.3e-4 | 2.8e-5 | 3.8e-5 |
| 12 | 0.37 | 0.014 | 0.219 | 1.5e-4 | −2.2e-5 | 0.041 | −4.0e-3 | 9.1e-5 | 1.8e-5 | 4.0e-5 |

**Share of proposals at K ≥ 4 whose entrant is at the floor:** 0.455 (normal), 0.492 (laplace), 0.505 (t₃). At
K = 4–8 it is 0.58–0.73.

**The brief's w-adjusted formula** is ½α(2−α)σ²[(1 − Σω²_new) − (1 − Σω²_old)]. It is in the CSV (`gain_w`) and the
figure.
- **What it measures.** It is the capital-weighted average of the members' gains, not an incumbent's.
- **Small w.** It agrees with the incumbent's gain to first order in w. For floor entrants it lies between the
  incumbent gain and the equal-weight gain (normal, K = 5: 1.5e-4).
- **Large w.** It turns negative once w exceeds about 2/K, while every member's own gain stays positive. So for
  the w ≥ 0.2 group the incumbent column is the relevant one.

### What this says about the hypothesis

**Recruit size explains most of the gap between the equal-weight formula and the binding gain, for the floor
entrants that make up the majority of proposals.**

- **Most proposals bring a floor entrant.** It holds a small share of the prospective unit: w̄ = 0.06–0.07 at
  K = 5 against 1/K = 0.2, and 0.02–0.03 at K = 8.
- **This shrinks the incumbents' gain.** An incumbent's gain from it is 36–37% of the equal-weight formula at
  K = 5 and 23–26% at K = 8. As a multiple of the marginal cost at K = 5:

  | | normal | laplace | t₃ |
  |---|---|---|---|
  | incumbent gain, floor entrant | 3.1× | 5.9× | 6.5× |
  | equal-weight formula | 8.4× | 16× | 18× |

- **Where the incumbent's net margin reaches zero:** K ≈ 7 for normal, between 8 and 9 for laplace and t₃. The
  equal-weight formula stays above the marginal cost up to K = 12 in all three.

**The binding mean for floor entrants follows that margin, shifted down.**
- **Normal and laplace.** It lies 2–5e-5 below the margin at every K ≥ 5. It crosses zero one or two members
  earlier: between K = 5 and 6 for normal, between 6 and 7 for laplace.
- The shift is the unanimity term C37 identified: the minimum over K noisy estimates sits below the expected
  margin.
- **So for normal and laplace** the zero crossing "from the fifth member on" decomposes into:
  - the small entrant, which brings the analytical margin to zero at K ≈ 7–9;
  - the minimum over noisy estimates, which moves the crossing two members earlier.
- **t₃.** The binding mean for floor entrants stays positive (1.5e-4 at K = 12) while acceptance falls to 0.22.
  With t₃ windows the mean is set by the rare large estimates. The mean does not show where the bulk of the
  distribution sits, and the counters store only mean and s.d.

**Entrants with w ≥ 0.2 do not fit the hypothesis.**
- They are 8–21% of proposals at K = 4–10, with w̄ ≈ 0.33–0.37.
- Their binding mean is strongly negative from K = 6 (normal, laplace) and K = 4 (t₃), and acceptance falls to
  1–4% at K = 12.
- The incumbent's analytical gain from such an entrant is *larger* than from a floor entrant. So their low
  binding gain is not a lack of diversification for the incumbents.
- **Likely type.** With w ≈ 1/3 at K ≥ 6, the entrant side holds the capital of two or more average members, so
  these are presumably mostly unit + unit proposals. There, members compare the proposal with the gain of the
  unit they already have.
- **Not verified.** The counters do not separate proposal types, so this is an inference.

**Verdict.** The hypothesis holds for the median proposal, which has a floor entrant: the incumbents' gain from
it is a third to a quarter of the equal-weight formula. It does not account for the low acceptance of large
entrants at K ≥ 6.

## 3. Size of the second-order approximation (Task 3)

**Design.**
- On each of the 27 pilot D traces, 2,000 random decision points: a standalone firm and a prospective unit of K − 1
  standalone partners from other markets, K ∈ {2, 3, 5, 8} nested.
- Δ̃ (moment) and Δ̂ (loggain replay) are computed with the model's own functions on the same 50-period window
  (`analysis/approximation_error.py`).
- **Inputs.** The windows are rebuilt from the trace: log-share changes net of the floor jump, and shares as
  sizes. They match the model's relative returns up to the market's renormalisation term (`deviations.md`).
- To separate the expansion's omitted terms from everything else, both are recomputed with the size weights frozen
  at the window's first period. The difference splits into:

  | term | definition | what it is |
  |---|---|---|
  | higher-order | Δ̃_fixed − Δ̂_fixed | the omitted terms proper |
  | replay weight term | Δ̂_fixed − Δ̂ | minus the replay's first-order term (1/h) Σ_t α Σ_j c_jt (u_jt − u_it), with the in-window weights |
  | moment weight term | Δ̃ − Δ̃_fixed | the weights' effect inside the moment rule |

Output: `pilot_d/approximation_error.csv`, by family × K pooled over α and reps (18,000 draws per row), and by
family × α × K (6,000). The table shows medians of each term as a share of Δ̂; the higher-order column is a share
of Δ̂_fixed.

| family | K | (Δ̃ − Δ̂)/Δ̂: median [5%, 95%] | higher-order, median [5%, 95%] | replay weight term | moment weight term | sign differs |
|---|---|---|---|---|---|---|
| normal | 2 | 0.27 [−0.32, 1.78] | 0.015 [−0.024, 0.049] | 0.22 | −0.003 | 2.3% |
| normal | 3 | 0.30 [−0.20, 1.45] | 0.015 [−0.017, 0.043] | 0.26 | −0.003 | 1.5% |
| normal | 5 | 0.33 [−0.13, 1.41] | 0.016 [−0.013, 0.041] | 0.30 | 0.000 | 1.5% |
| normal | 8 | 0.38 [−0.12, 1.59] | 0.017 [−0.012, 0.047] | 0.35 | 0.003 | 1.7% |
| laplace | 2 | 0.22 [−0.25, 1.56] | 0.010 [−0.073, 0.077] | 0.18 | 0.000 | 1.5% |
| laplace | 3 | 0.25 [−0.17, 1.22] | 0.010 [−0.064, 0.066] | 0.22 | −0.001 | 1.0% |
| laplace | 5 | 0.29 [−0.09, 1.13] | 0.008 [−0.058, 0.058] | 0.27 | 0.000 | 0.9% |
| laplace | 8 | 0.33 [−0.04, 1.21] | 0.008 [−0.054, 0.056] | 0.32 | 0.001 | 0.8% |
| t₃ | 2 | 0.20 [−0.32, 1.52] | 0.009 [−0.188, 0.192] | 0.17 | −0.002 | 1.4% |
| t₃ | 3 | 0.23 [−0.24, 1.34] | 0.012 [−0.179, 0.253] | 0.20 | −0.004 | 0.9% |
| t₃ | 5 | 0.27 [−0.18, 1.33] | 0.014 [−0.186, 0.309] | 0.23 | −0.002 | 0.7% |
| t₃ | 8 | 0.32 [−0.16, 1.45] | 0.017 [−0.191, 0.351] | 0.27 | −0.002 | 0.8% |

**The omitted higher-order terms are small relative to the gain for normal and laplace, and heavy-tailed for t₃.**

| family | median share of the gain | 90% of windows within | mean |
|---|---|---|---|
| normal | 1.5–1.7% | −2.4% to +4.9% | – |
| laplace | 0.8–1.0% | −7.3% to +7.7% | – |
| t₃ | 0.9–1.7% | −19% to +35% | at α = 0.3, 1.4–2.3e-3, about half the mean difference |

For t₃ the mean is set by rare windows with extreme draws, where the expansion fails.

**The bulk of the gap is not higher-order.**
- Δ̃ exceeds Δ̂ by a median 20–38% of Δ̂ (36–51% at normal α = 0.3).
- Almost all of it is the replay's first-order term (17–35% of Δ̂ pooled, up to 48% at normal α = 0.3, K = 8).
- The moment rule drops that term as mean zero, because the weights are set before the period's return. In a
  window where each firm's return is demeaned, it is not zero. A firm's weight is high after its returns were
  above its window mean, and the demeaned window then needs below-mean returns. So the in-window covariance
  between weights and returns is negative, and Δ̂ comes out below what the same window gives with weights fixed
  in advance.
- The moment rule's own weight term is negligible (|median| ≤ 0.8% of Δ̂).
- Pooled over α, the two estimates disagree in sign in only 0.7–2.3% of windows (at most 3.5% in any α cell),
  because the shift is mostly a level shift.

**Implication for C37.**
- The verifiability component (c) of the consent gap, K_moment − K_loggain, is partly this finite-window downward
  bias of Δ̂, not only its sampling noise.
- On the same windows the loggain estimate is lower than the moment estimate: median Δ̂ over median Δ̃ is
  0.71–0.81 pooled over α (0.66–0.86 by α).

## 4. Fixed α = 0.9 against the endogenous-α block (Task 4)

**Design.** Block `alpha-0.9`, ids 1995–2009: the main power-law cells at α = 0.9, 5 reps, moment rule, seeds of
the main cell. Table: medians over the 5 reps from `pilot_d/tidy.csv` (`pilot_d/fixed_alpha_09.csv`). The pooled
share is the capital share in units.

| family | cell | K | pooled share | Hill | floor exits per period | adopted α | acceptance |
|---|---|---|---|---|---|---|---|
| normal | main α = 0.5 | 7.70 | 0.956 | 1.342 | 15.59 | – | 0.269 |
| normal | fixed α = 0.9 | 15.99 | 0.980 | 1.978 | 2.44 | – | 0.155 |
| normal | endogenous α | 16.93 | 0.977 | 2.052 | 2.26 | 0.90 | 0.151 |
| laplace | main α = 0.5 | 7.76 | 0.934 | 1.344 | 24.49 | – | 0.379 |
| laplace | fixed α = 0.9 | 17.20 | 0.966 | 1.942 | 4.90 | – | 0.215 |
| laplace | endogenous α | 18.00 | 0.968 | 2.067 | 4.42 | 0.90 | 0.205 |
| t₃ | main α = 0.5 | 6.78 | 0.920 | 1.221 | 31.15 | – | 0.409 |
| t₃ | fixed α = 0.9 | 13.94 | 0.950 | 1.783 | 10.55 | – | 0.298 |
| t₃ | endogenous α | 14.28 | 0.948 | 1.826 | 11.23 | 0.90 | 0.302 |

**A fixed α of 0.9 reproduces the endogenous-α block closely.**

| | fixed α = 0.9 against endogenous α |
|---|---|
| K | 2–6% lower |
| Hill | 0.04–0.13 lower |
| pooled share | within 0.003 |
| floor exits | within 0.7 per period, in either direction |
| acceptance | within 0.01 |

So the endogenous block's outcomes are essentially those of its adopted level. Letting each unit choose and
renegotiate α adds little on top: slightly larger units and slightly higher concentration.

**Against main α = 0.5,** going to α ≈ 0.9:
- doubles K;
- raises Hill by 0.56–0.64;
- cuts floor exits by a factor of 3 (t₃) to 6 (normal).
