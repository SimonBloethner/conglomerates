# C35 note: what binds scope besides attrition

Branch `referee-fixes`, from C34 (`46cc0a5`). The model does not change. C35b adds diagnostic counters to the
merger kernel, and every existing output stays byte-identical.

Commits:
- C35a: decomposition of the owner-return gap;
- C35b: kernel diagnostics and the identity check;
- C35c: the cost-level-high block;
- a fourth commit with the run outputs.

Details are in `c35_decomposition.md` and `c35_acceptance.md`, in the "Cost level high" section of
`pilot_c_summary.md`, and in the C35 section of `deviations.md`.

Runs were on Festus in `/scratch/bt307958/c35/IOxEE`. The tree is copied to
`/groups/m-larch/bt307958/IOxEE_c35` and verified by checksum.

## 1. Decomposition of the owner-return gap (27 traces)

Gap = next-period owner return of members minus standalones, within (market, size decile), weighted by member
firm-periods. Values are means over 3 reps. The split is exact (residual ≤ 2e-18):
- **ordinary periods** A = (1 − p_s)(m_m,nf − m_s,nf);
- **ruin protection** B + C = p_m(m_m,f − m_m,nf) − p_s(m_s,f − m_m,nf).

| family | α | total gap | no-floor gap | floor gap | A ordinary | B + C ruin | share ordinary |
|---|---|---|---|---|---|---|---|
| normal | 0.05 | 0.00054 | -0.00032 | -0.00016 | -0.00029 | 0.00084 | -0.63 |
| normal | 0.1 | 0.00131 | -0.00000 | 0.00627 | 0.00004 | 0.00127 | 0.02 |
| normal | 0.3 | 0.00427 | 0.00160 | 0.02680 | 0.00177 | 0.00249 | 0.41 |
| laplace | 0.05 | 0.00103 | -0.00061 | 0.00008 | -0.00056 | 0.00159 | -0.56 |
| laplace | 0.1 | 0.00272 | 0.00022 | 0.00898 | 0.00027 | 0.00245 | 0.10 |
| laplace | 0.3 | 0.00815 | 0.00292 | 0.04137 | 0.00308 | 0.00507 | 0.38 |
| t3 | 0.05 | 0.00266 | -0.00065 | 0.03258 | -0.00059 | 0.00325 | -0.23 |
| t3 | 0.1 | 0.00452 | -0.00042 | 0.04517 | -0.00035 | 0.00487 | -0.08 |
| t3 | 0.3 | 0.01136 | 0.00190 | 0.06972 | 0.00205 | 0.00932 | 0.18 |

## 2. Acceptance by K against marginal gain and marginal cost (α = 0.3, p = 0.05)

- **Acceptance** = post-burn-in mergers / proposals at merged size K, pooled over 5 reps.
- **ΔG** is the Section 2 marginal gain of the K-th member, ½α(2 − α)σ²[1/(K−1) − 1/K], at the median market σ
  (normal 0.148, laplace 0.205, t₃ 0.217).
- **ΔΦ** = Φ(K) − Φ(K−1), the same for all families.
- **Binding gain** = min_i [Δ̂_i(Ĉ) − max(0, Δ̂_i(C_i))], median [p25, p75].
- The brief's (K → K+1) columns, p = 1 and all families in full are in `c35_acceptance.md` and
  `pilot_c/acceptance_by_K*.csv`.

| K | acc. normal | acc. laplace | acc. t₃ | ΔG normal | ΔG laplace | ΔG t₃ | ΔΦ | binding gain, normal |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.919 | 0.965 | 0.970 | 2.8e-03 | 5.4e-03 | 6.0e-03 | 2.6e-05 | 1.6e-03 [7.3e-04, 2.6e-03] |
| 3 | 0.553 | 0.593 | 0.569 | 9.3e-04 | 1.8e-03 | 2.0e-03 | 2.9e-05 | 5.6e-05 [-1.8e-04, 4.7e-04] |
| 4 | 0.375 | 0.409 | 0.393 | 4.6e-04 | 9.0e-04 | 1.0e-03 | 3.1e-05 | -7.0e-05 [-2.8e-04, 1.2e-04] |
| 5 | 0.275 | 0.313 | 0.300 | 2.8e-04 | 5.4e-04 | 6.0e-04 | 3.3e-05 | -9.7e-05 [-3.2e-04, 2.2e-05] |
| 6 | 0.210 | 0.248 | 0.242 | 1.9e-04 | 3.6e-04 | 4.0e-04 | 3.4e-05 | -1.1e-04 [-3.7e-04, -1.5e-05] |
| 8 | 0.129 | 0.169 | 0.163 | 9.9e-05 | 1.9e-04 | 2.2e-04 | 3.6e-05 | -1.5e-04 [-5.4e-04, -4.0e-05] |
| 10 | 0.073 | 0.109 | 0.114 | 6.2e-05 | 1.2e-04 | 1.3e-04 | 3.8e-05 | -2.9e-04 [-8.0e-04, -6.8e-05] |
| 12 | 0.038 | 0.069 | 0.072 | 4.2e-05 | 8.2e-05 | 9.1e-05 | 4.0e-05 | -5.6e-04 [-1.0e-03, -1.4e-04] |
| 15 | 0.009 | 0.029 | 0.032 | 2.6e-05 | 5.1e-05 | 5.7e-05 | 4.2e-05 | -8.4e-04 [-1.2e-03, -5.1e-04] |
| 20 | 0.000 | 0.004 | 0.000 | 1.5e-05 | 2.8e-05 | 3.2e-05 | 4.4e-05 | -1.0e-03 [-1.4e-03, -7.8e-04] |

## 3. Cost-level-high block

Power law Φ(K) = c0·K^1.2 with c0 × {5, 10, 20}; laplace and t₃; α ∈ {0.1, 0.3, 0.5}; 5 reps; scenario ids
1800–1889. ×1 is the main power-law cell. Values are medians over reps. K* is from `analytics/benchmarks.csv`
(M = 50, IQR 0.2).

| family | × c0 | α | K | K* | K/K* | pooled share | acceptance | mergers / period | floor exits / period |
|---|---|---|---|---|---|---|---|---|---|
| laplace | 1 | 0.1 | 3.005 | 10 | 0.300 | 0.634 | 0.413 | 48.57 | 46.77 |
| laplace | 1 | 0.3 | 3.999 | 16 | 0.250 | 0.782 | 0.361 | 37.76 | 36.88 |
| laplace | 1 | 0.5 | 5.948 | 20 | 0.297 | 0.906 | 0.321 | 25.24 | 25.05 |
| laplace | 5 | 0.1 | 2.480 | 5 | 0.496 | 0.438 | 0.363 | 44.07 | 41.09 |
| laplace | 5 | 0.3 | 3.138 | 8 | 0.392 | 0.636 | 0.314 | 36.09 | 34.18 |
| laplace | 5 | 0.5 | 4.129 | 9 | 0.459 | 0.815 | 0.244 | 24.55 | 23.97 |
| laplace | 10 | 0.1 | 2.276 | 4 | 0.569 | 0.337 | 0.334 | 41.06 | 37.99 |
| laplace | 10 | 0.3 | 2.760 | 6 | 0.460 | 0.536 | 0.311 | 36.96 | 34.22 |
| laplace | 10 | 0.5 | 3.441 | 7 | 0.492 | 0.699 | 0.241 | 26.75 | 25.46 |
| laplace | 20 | 0.1 | 2.120 | 3 | 0.707 | 0.208 | 0.257 | 31.87 | 29.64 |
| laplace | 20 | 0.3 | 2.452 | 4 | 0.613 | 0.410 | 0.306 | 37.31 | 33.80 |
| laplace | 20 | 0.5 | 2.847 | 5 | 0.569 | 0.566 | 0.251 | 29.64 | 27.22 |
| t3 | 1 | 0.1 | 2.866 | 20 | 0.143 | 0.627 | 0.452 | 53.69 | 52.13 |
| t3 | 1 | 0.3 | 3.667 | 47 | 0.078 | 0.773 | 0.403 | 44.12 | 43.18 |
| t3 | 1 | 0.5 | 5.063 | 50 | 0.101 | 0.884 | 0.359 | 32.60 | 32.28 |
| t3 | 5 | 0.1 | 2.491 | 7 | 0.356 | 0.496 | 0.425 | 51.84 | 49.71 |
| t3 | 5 | 0.3 | 3.042 | 10 | 0.304 | 0.662 | 0.379 | 44.13 | 42.69 |
| t3 | 5 | 0.5 | 3.881 | 13 | 0.299 | 0.814 | 0.319 | 33.80 | 33.25 |
| t3 | 10 | 0.1 | 2.318 | 4 | 0.579 | 0.408 | 0.400 | 49.12 | 46.72 |
| t3 | 10 | 0.3 | 2.766 | 7 | 0.395 | 0.585 | 0.373 | 44.48 | 42.57 |
| t3 | 10 | 0.5 | 3.343 | 9 | 0.371 | 0.742 | 0.310 | 34.98 | 34.02 |
| t3 | 20 | 0.1 | 2.161 | 3 | 0.720 | 0.294 | 0.341 | 42.12 | 40.00 |
| t3 | 20 | 0.3 | 2.496 | 5 | 0.499 | 0.477 | 0.368 | 44.61 | 42.05 |
| t3 | 20 | 0.5 | 2.884 | 6 | 0.481 | 0.631 | 0.313 | 36.81 | 35.16 |

## 4. Identity report

**Kernel diagnostics (C35b).** `analytics/identity_c35.py` compares C34 and C35 result pickles field by field,
byte for byte, leaving out only the C35 keys. It also compares the tidy rows as text.
- **Runs compared:** 146 at full T = 11,000. These are one scenario of each of the 56 C34 cells that the Task 2
  reruns do not cover, plus the 90 Task 2 reruns (main power-law and search p = 1, three families,
  α ∈ {0.1, 0.3, 0.5}, 5 reps). Every block and every rule is covered, the replay rule and endogenous α included.
- **Result:** 146/146 identical (fields and tidy row).
- **Counters:** consistent in 146/146. Per period, proposals_by_K[2] = solo+solo and Σ_{K≥3} = solo+unit +
  unit+unit, and the same for mergers. The post-burn-in sums match, and the histogram holds one count per
  post-burn-in proposal under loggain and none under replay.
- **Runtime:** median ratio C35/C34 = 1.04.
- **Results directory:** reruns are in `pilot_c/results_c35/`; the C34 files are untouched.

**Traces (C35a).** The 27 traces' per-step floor-hit totals equal the C34 pickles, and the floor hits align with
the recapitalisation jumps in 27/27.

**Cost-level-high (C35c).**
- `tidy.csv`: the first 1,800 rows (and header) are byte-identical to C34.
- `medians.csv`: all old rows are kept, plus 18 new ones.
- `scenarios.json`: scenarios 0–1799 are unchanged.

**Tests.** `pytest tests/` on Festus: 226 passed, 0 failed.

## 5. Answers

**What share of the gain is ordinary-period growth?**
- **At α = 0.3:** 41% (normal), 38% (laplace) and 18% (t₃). The other 59–82% comes from members reaching the floor
  less often and losing less when they do. This is below the brief's expectation of about one half (normal) and
  one third (t₃), and the gap is largest under t₃.
- **At α ≤ 0.1:** the ordinary-period component is about zero (normal and laplace at 0.1: 0.02 and 0.10) or
  negative (all families at 0.05, t₃ at 0.1). In ordinary periods members then grow a little *less* than
  standalones in the same cell, and the whole gain is ruin protection.
- **Caveat:** the split is exact accounting on a sample selected on the outcome. The floor removes more of the
  standalones' lower tail, so A is a conservative measure of the ordinary-period component.

**Does consent bind at large K, and why?**
- **Acceptance falls monotonically with K**: from 0.92–0.98 at K = 2 to 0.38–0.42 at K = 4, 0.13–0.17 at K = 8 and
  about 0 by K = 20. The profile is the same at p = 0.05 and p = 1.
- **Not because the marginal member loses in expectation.** By the Section 2 formula, the K-th member's marginal
  gain exceeds the marginal cost up to K ≈ 12 (normal), 16 (laplace) and 17 (t₃). At K = 4–8 it is 3–30 times
  the cost.
- **Because the gains are small next to the noise.** In this range the binding member's estimated gain is centred
  just below zero (−4e-5 to −3e-4), with an interquartile range (2–14e-4) as wide as the marginal gain itself. The
  50-period window cannot resolve a gain of order 1e-4, and unanimity takes the minimum of K such estimates, which
  falls as K grows.
- **So consent binds where scope actually forms (K ≈ 3–8), whatever the cost.** Beyond K ≈ 11–12 the binding gains
  are clearly negative (median −3e-4 at K = 12 down to −2.6e-3 at K = 25, IQR below zero): the minimum over many members is lower, and past
  K ≈ 12–17 the marginal gain falls below the marginal cost.

**Where does cost take over?**
- **No threshold:** scope falls by about 0.35 in mean K per doubling of c0, from ×0.5 to ×20 (laplace α = 0.3:
  4.33, 4.00, 3.67, 3.14, 2.76, 2.45 at ×0.5, 1, 2, 5, 10, 20). Cost acts at the margin even at ×1, because
  the binding gains sit near zero, so any shift of their mean moves acceptance. At ×1, though, ΔΦ is only a
  tenth of ΔG at K = 4.
- **Cost does not take over by ×20.** K\* falls faster than K: K/K\* rises from 0.25 (laplace) and 0.08 (t₃) at ×1
  to 0.61 and 0.50 at ×20 (α = 0.3). At ×20, K\* = 4–5 matches the scope that forms at ×1, and the analytical
  marginal cost of the 4th–5th member is close to its marginal gain. Yet K is 2.45–2.50, and 2.85–2.88 at α = 0.5
  against K\* = 5–6. K never settles at K\*.
- **So up to ×20 (Φ(4) ≈ 2e-3, a cost/gain ratio of roughly 20–60%), cost limits scope only together with
  attrition and consent.** It would be the sole constraint only at a level where K\* falls below the scope that
  attrition and consent allow. Even at ×20 that is not the case.
