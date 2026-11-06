# C35 Task 2 — acceptance by unit size K and the binding member's gain

## Sources and definitions

**Sources.**
- `pilot_c/acceptance_by_K.csv` has one row per run and K.
- `pilot_c/acceptance_by_K_pooled.csv` has the same data pooled over the 5 reps.
- Both are produced by `analytics/acceptance_c35.py`, from the C35 kernel counters (`MODEL_FLAGS.md`, "Acceptance
  by K").
- The figure is `diagnostics/pilot_c_acceptance_by_K.png`.
- Both CSVs include: the main block (p = 0.05) and the search block at p = 1, for power-law cost, the three
  families and α ∈ {0.1, 0.3, 0.5}; the cost-level-high block; and three search-block runs at p = 0.2, α = 0.3
  from the identity sample.

**Definitions.** All counts are post-burn-in, pooled over the 5 reps.
- **K** is the size of the proposed merged set.
- **acceptance** = mergers / proposals at that K.
- **binding gain** = min over members i of Δ̂_i(Ĉ) − max{0, Δ̂_i(C_i)}. It is positive if and only if the proposal
  is accepted. Its median and interquartile range come from the 1e-4-wide histogram, interpolated within bins.
- **ΔG into K** = ½α(2 − α)σ²[1/(K−1) − 1/K], the Section 2 marginal gain of the K-th member. σ is the median over
  markets of the s.d. of the log shock (normal 0.148, laplace 0.205, t₃ 0.217 at p = 0.05).
- **ΔΦ into K** = Φ(K) − Φ(K−1).
- **ΔG next** and **ΔΦ next** are the brief's columns, for the (K+1)-th member.
- Read the acceptance at size K against the "into K" columns. A solo + unit proposal of size K adds the K-th
  member to a unit of K − 1. A unit + unit proposal adds several members at once.

## Results by family and search probability (α = 0.3)

**normal, α = 0.3, p = 0.05** (5 runs, σ_median = 0.148)

| K | proposals | acceptance | binding gain median [p25, p75] | share > 0 | ΔG into K | ΔΦ into K | ΔG next | ΔΦ next |
|---|---|---|---|---|---|---|---|---|
| 2 | 172680 | 0.919 | 1.6e-03 [7.3e-04, 2.6e-03] | 0.919 | 2.8e-03 | 2.6e-05 | 9.3e-04 | 2.9e-05 |
| 3 | 111008 | 0.553 | 5.6e-05 [-1.8e-04, 4.7e-04] | 0.553 | 9.3e-04 | 2.9e-05 | 4.6e-04 | 3.1e-05 |
| 4 | 113850 | 0.375 | -7.0e-05 [-2.8e-04, 1.2e-04] | 0.375 | 4.6e-04 | 3.1e-05 | 2.8e-04 | 3.3e-05 |
| 5 | 127760 | 0.275 | -9.7e-05 [-3.2e-04, 2.2e-05] | 0.275 | 2.8e-04 | 3.3e-05 | 1.9e-04 | 3.4e-05 |
| 6 | 140947 | 0.210 | -1.1e-04 [-3.7e-04, -1.5e-05] | 0.210 | 1.9e-04 | 3.4e-05 | 1.3e-04 | 3.5e-05 |
| 7 | 148752 | 0.165 | -1.2e-04 [-4.3e-04, -2.9e-05] | 0.165 | 1.3e-04 | 3.5e-05 | 9.9e-05 | 3.6e-05 |
| 8 | 146505 | 0.129 | -1.5e-04 [-5.4e-04, -4.0e-05] | 0.129 | 9.9e-05 | 3.6e-05 | 7.7e-05 | 3.7e-05 |
| 9 | 132093 | 0.099 | -1.9e-04 [-6.7e-04, -5.2e-05] | 0.099 | 7.7e-05 | 3.7e-05 | 6.2e-05 | 3.8e-05 |
| 10 | 110618 | 0.073 | -2.9e-04 [-8.0e-04, -6.8e-05] | 0.073 | 6.2e-05 | 3.8e-05 | 5.1e-05 | 3.9e-05 |
| 11 | 87478 | 0.051 | -4.3e-04 [-9.2e-04, -9.1e-05] | 0.051 | 5.1e-05 | 3.9e-05 | 4.2e-05 | 4.0e-05 |
| 12 | 65707 | 0.038 | -5.6e-04 [-1.0e-03, -1.4e-04] | 0.038 | 4.2e-05 | 4.0e-05 | 3.6e-05 | 4.0e-05 |
| 15 | 20634 | 0.009 | -8.4e-04 [-1.2e-03, -5.1e-04] | 0.009 | 2.6e-05 | 4.2e-05 | 2.3e-05 | 4.2e-05 |
| 20 | 1041 | 0.000 | -1.0e-03 [-1.4e-03, -7.8e-04] | 0.000 | 1.5e-05 | 4.4e-05 | 1.3e-05 | 4.5e-05 |
| 25 | 10 | 0.000 | -1.3e-03 [-1.4e-03, -9.5e-04] | 0.000 | 9.5e-06 | 4.6e-05 | 8.8e-06 | 4.7e-05 |

**normal, α = 0.3, p = 1** (5 runs, σ_median = 0.150)

| K | proposals | acceptance | binding gain median [p25, p75] | share > 0 | ΔG into K | ΔΦ into K | ΔG next | ΔΦ next |
|---|---|---|---|---|---|---|---|---|
| 2 | 259919 | 0.925 | 1.8e-03 [1.2e-03, 2.5e-03] | 0.925 | 2.9e-03 | 2.6e-05 | 9.6e-04 | 2.9e-05 |
| 3 | 145048 | 0.608 | 1.5e-04 [-7.8e-05, 6.3e-04] | 0.608 | 9.6e-04 | 2.9e-05 | 4.8e-04 | 3.1e-05 |
| 4 | 157770 | 0.404 | -4.0e-05 [-1.9e-04, 1.8e-04] | 0.404 | 4.8e-04 | 3.1e-05 | 2.9e-04 | 3.3e-05 |
| 5 | 240195 | 0.279 | -7.4e-05 [-2.2e-04, 2.7e-05] | 0.279 | 2.9e-04 | 3.3e-05 | 1.9e-04 | 3.4e-05 |
| 6 | 399410 | 0.210 | -8.1e-05 [-2.2e-04, -1.1e-05] | 0.210 | 1.9e-04 | 3.4e-05 | 1.4e-04 | 3.5e-05 |
| 7 | 633911 | 0.167 | -8.3e-05 [-2.2e-04, -2.1e-05] | 0.167 | 1.4e-04 | 3.5e-05 | 1.0e-04 | 3.6e-05 |
| 8 | 888171 | 0.135 | -8.9e-05 [-2.5e-04, -2.8e-05] | 0.135 | 1.0e-04 | 3.6e-05 | 8.0e-05 | 3.7e-05 |
| 9 | 1100110 | 0.109 | -9.7e-05 [-3.4e-04, -3.5e-05] | 0.109 | 8.0e-05 | 3.7e-05 | 6.4e-05 | 3.8e-05 |
| 10 | 1232093 | 0.087 | -1.3e-04 [-5.7e-04, -4.4e-05] | 0.087 | 6.4e-05 | 3.8e-05 | 5.2e-05 | 3.9e-05 |
| 11 | 1295349 | 0.066 | -2.0e-04 [-8.0e-04, -5.9e-05] | 0.066 | 5.2e-05 | 3.9e-05 | 4.4e-05 | 4.0e-05 |
| 12 | 1295642 | 0.047 | -4.2e-04 [-9.9e-04, -8.2e-05] | 0.047 | 4.4e-05 | 4.0e-05 | 3.7e-05 | 4.0e-05 |
| 15 | 1081880 | 0.012 | -8.7e-04 [-1.3e-03, -5.1e-04] | 0.012 | 2.7e-05 | 4.2e-05 | 2.4e-05 | 4.2e-05 |
| 20 | 283658 | 0.001 | -1.1e-03 [-1.5e-03, -8.2e-04] | 0.001 | 1.5e-05 | 4.4e-05 | 1.4e-05 | 4.5e-05 |
| 25 | 13832 | 0.000 | -1.3e-03 [-1.7e-03, -9.7e-04] | 0.000 | 9.6e-06 | 4.6e-05 | 8.8e-06 | 4.7e-05 |

**laplace, α = 0.3, p = 0.05** (5 runs, σ_median = 0.205)

| K | proposals | acceptance | binding gain median [p25, p75] | share > 0 | ΔG into K | ΔΦ into K | ΔG next | ΔΦ next |
|---|---|---|---|---|---|---|---|---|
| 2 | 246897 | 0.965 | 3.5e-03 [1.9e-03, 5.2e-03] | 0.965 | 5.4e-03 | 2.6e-05 | 1.8e-03 | 2.9e-05 |
| 3 | 148739 | 0.593 | 1.7e-04 [-2.8e-04, 1.0e-03] | 0.593 | 1.8e-03 | 2.9e-05 | 9.0e-04 | 3.1e-05 |
| 4 | 141933 | 0.409 | -8.3e-05 [-4.7e-04, 2.6e-04] | 0.409 | 9.0e-04 | 3.1e-05 | 5.4e-04 | 3.3e-05 |
| 5 | 146316 | 0.313 | -1.4e-04 [-5.3e-04, 7.2e-05] | 0.313 | 5.4e-04 | 3.3e-05 | 3.6e-04 | 3.4e-05 |
| 6 | 149277 | 0.248 | -1.6e-04 [-6.0e-04, -9.4e-07] | 0.248 | 3.6e-04 | 3.4e-05 | 2.6e-04 | 3.5e-05 |
| 7 | 145749 | 0.202 | -1.8e-04 [-7.2e-04, -2.4e-05] | 0.202 | 2.6e-04 | 3.5e-05 | 1.9e-04 | 3.6e-05 |
| 8 | 135349 | 0.169 | -1.9e-04 [-8.9e-04, -3.8e-05] | 0.169 | 1.9e-04 | 3.6e-05 | 1.5e-04 | 3.7e-05 |
| 9 | 118288 | 0.137 | -2.4e-04 [-1.1e-03, -5.2e-05] | 0.137 | 1.5e-04 | 3.7e-05 | 1.2e-04 | 3.8e-05 |
| 10 | 97593 | 0.109 | -3.3e-04 [-1.3e-03, -6.7e-05] | 0.109 | 1.2e-04 | 3.8e-05 | 9.8e-05 | 3.9e-05 |
| 11 | 74379 | 0.087 | -5.2e-04 [-1.5e-03, -8.6e-05] | 0.087 | 9.8e-05 | 3.9e-05 | 8.2e-05 | 4.0e-05 |
| 12 | 55435 | 0.069 | -7.2e-04 [-1.7e-03, -1.1e-04] | 0.069 | 8.2e-05 | 4.0e-05 | 6.9e-05 | 4.0e-05 |
| 15 | 16717 | 0.029 | -1.2e-03 [-2.1e-03, -4.9e-04] | 0.029 | 5.1e-05 | 4.2e-05 | 4.5e-05 | 4.2e-05 |
| 20 | 935 | 0.004 | -1.7e-03 [-2.4e-03, -1.1e-03] | 0.004 | 2.8e-05 | 4.4e-05 | 2.6e-05 | 4.5e-05 |
| 25 | 15 | 0.000 | -1.6e-03 [-2.4e-03, -1.2e-03] | 0.000 | 1.8e-05 | 4.6e-05 | 1.7e-05 | 4.7e-05 |

**laplace, α = 0.3, p = 1** (5 runs, σ_median = 0.202)

| K | proposals | acceptance | binding gain median [p25, p75] | share > 0 | ΔG into K | ΔΦ into K | ΔG next | ΔΦ next |
|---|---|---|---|---|---|---|---|---|
| 2 | 373897 | 0.971 | 3.2e-03 [2.1e-03, 4.6e-03] | 0.971 | 5.3e-03 | 2.6e-05 | 1.8e-03 | 2.9e-05 |
| 3 | 210133 | 0.638 | 3.2e-04 [-1.5e-04, 1.2e-03] | 0.638 | 1.8e-03 | 2.9e-05 | 8.8e-04 | 3.1e-05 |
| 4 | 222304 | 0.421 | -6.0e-05 [-4.0e-04, 3.4e-04] | 0.421 | 8.8e-04 | 3.1e-05 | 5.3e-04 | 3.3e-05 |
| 5 | 306466 | 0.302 | -1.2e-04 [-4.3e-04, 6.6e-05] | 0.302 | 5.3e-04 | 3.3e-05 | 3.5e-04 | 3.4e-05 |
| 6 | 460472 | 0.236 | -1.4e-04 [-4.1e-04, -6.6e-06] | 0.236 | 3.5e-04 | 3.4e-05 | 2.5e-04 | 3.5e-05 |
| 7 | 665807 | 0.193 | -1.4e-04 [-3.9e-04, -2.3e-05] | 0.193 | 2.5e-04 | 3.5e-05 | 1.9e-04 | 3.6e-05 |
| 8 | 892081 | 0.163 | -1.4e-04 [-4.1e-04, -3.3e-05] | 0.163 | 1.9e-04 | 3.6e-05 | 1.5e-04 | 3.7e-05 |
| 9 | 1095911 | 0.138 | -1.5e-04 [-5.1e-04, -4.1e-05] | 0.138 | 1.5e-04 | 3.7e-05 | 1.2e-04 | 3.8e-05 |
| 10 | 1238703 | 0.116 | -1.7e-04 [-7.5e-04, -5.1e-05] | 0.116 | 1.2e-04 | 3.8e-05 | 9.6e-05 | 3.9e-05 |
| 11 | 1303235 | 0.095 | -2.2e-04 [-1.1e-03, -6.4e-05] | 0.095 | 9.6e-05 | 3.9e-05 | 8.0e-05 | 4.0e-05 |
| 12 | 1303218 | 0.075 | -3.4e-04 [-1.5e-03, -8.2e-05] | 0.075 | 8.0e-05 | 4.0e-05 | 6.7e-05 | 4.0e-05 |
| 15 | 1024002 | 0.030 | -1.2e-03 [-2.2e-03, -3.1e-04] | 0.030 | 5.0e-05 | 4.2e-05 | 4.4e-05 | 4.2e-05 |
| 20 | 297245 | 0.005 | -1.7e-03 [-2.5e-03, -1.1e-03] | 0.005 | 2.8e-05 | 4.4e-05 | 2.5e-05 | 4.5e-05 |
| 25 | 22059 | 0.001 | -1.9e-03 [-2.7e-03, -1.3e-03] | 0.001 | 1.8e-05 | 4.6e-05 | 1.6e-05 | 4.7e-05 |

**t3, α = 0.3, p = 0.05** (5 runs, σ_median = 0.217)

| K | proposals | acceptance | binding gain median [p25, p75] | share > 0 | ΔG into K | ΔΦ into K | ΔG next | ΔΦ next |
|---|---|---|---|---|---|---|---|---|
| 2 | 308316 | 0.970 | 4.0e-03 [2.3e-03, 6.6e-03] | 0.970 | 6.0e-03 | 2.6e-05 | 2.0e-03 | 2.9e-05 |
| 3 | 195833 | 0.569 | 1.5e-04 [-4.5e-04, 1.2e-03] | 0.569 | 2.0e-03 | 2.9e-05 | 1.0e-03 | 3.1e-05 |
| 4 | 180853 | 0.393 | -1.3e-04 [-7.0e-04, 2.9e-04] | 0.393 | 1.0e-03 | 3.1e-05 | 6.0e-04 | 3.3e-05 |
| 5 | 174660 | 0.300 | -2.0e-04 [-8.0e-04, 7.0e-05] | 0.300 | 6.0e-04 | 3.3e-05 | 4.0e-04 | 3.4e-05 |
| 6 | 165257 | 0.242 | -2.2e-04 [-9.2e-04, -6.0e-06] | 0.242 | 4.0e-04 | 3.4e-05 | 2.9e-04 | 3.5e-05 |
| 7 | 148813 | 0.196 | -2.6e-04 [-1.1e-03, -3.6e-05] | 0.196 | 2.9e-04 | 3.5e-05 | 2.2e-04 | 3.6e-05 |
| 8 | 125889 | 0.163 | -3.1e-04 [-1.4e-03, -5.6e-05] | 0.163 | 2.2e-04 | 3.6e-05 | 1.7e-04 | 3.7e-05 |
| 9 | 102679 | 0.135 | -3.9e-04 [-1.7e-03, -7.3e-05] | 0.135 | 1.7e-04 | 3.7e-05 | 1.3e-04 | 3.8e-05 |
| 10 | 78745 | 0.114 | -5.4e-04 [-1.9e-03, -8.9e-05] | 0.114 | 1.3e-04 | 3.8e-05 | 1.1e-04 | 3.9e-05 |
| 11 | 58207 | 0.091 | -7.4e-04 [-2.2e-03, -1.2e-04] | 0.091 | 1.1e-04 | 3.9e-05 | 9.1e-05 | 4.0e-05 |
| 12 | 39696 | 0.072 | -1.0e-03 [-2.4e-03, -1.6e-04] | 0.072 | 9.1e-05 | 4.0e-05 | 7.7e-05 | 4.0e-05 |
| 15 | 9744 | 0.032 | -1.6e-03 [-2.9e-03, -6.3e-04] | 0.032 | 5.7e-05 | 4.2e-05 | 5.0e-05 | 4.2e-05 |
| 20 | 376 | 0.000 | -2.1e-03 [-3.1e-03, -1.4e-03] | 0.000 | 3.2e-05 | 4.4e-05 | 2.9e-05 | 4.5e-05 |
| 25 | 5 | 0.000 | -2.2e-03 [-2.3e-03, -1.8e-03] | 0.000 | 2.0e-05 | 4.6e-05 | 1.9e-05 | 4.7e-05 |

**t3, α = 0.3, p = 1** (5 runs, σ_median = 0.229)

| K | proposals | acceptance | binding gain median [p25, p75] | share > 0 | ΔG into K | ΔΦ into K | ΔG next | ΔΦ next |
|---|---|---|---|---|---|---|---|---|
| 2 | 505018 | 0.981 | 3.9e-03 [2.6e-03, 5.8e-03] | 0.981 | 6.7e-03 | 2.6e-05 | 2.2e-03 | 2.9e-05 |
| 3 | 321070 | 0.595 | 2.8e-04 [-4.4e-04, 1.3e-03] | 0.595 | 2.2e-03 | 2.9e-05 | 1.1e-03 | 3.1e-05 |
| 4 | 347071 | 0.387 | -1.3e-04 [-7.5e-04, 3.5e-04] | 0.387 | 1.1e-03 | 3.1e-05 | 6.7e-04 | 3.3e-05 |
| 5 | 471513 | 0.278 | -2.2e-04 [-7.5e-04, 4.7e-05] | 0.278 | 6.7e-04 | 3.3e-05 | 4.5e-04 | 3.4e-05 |
| 6 | 661457 | 0.219 | -2.3e-04 [-7.1e-04, -2.1e-05] | 0.219 | 4.5e-04 | 3.4e-05 | 3.2e-04 | 3.5e-05 |
| 7 | 893428 | 0.180 | -2.3e-04 [-7.2e-04, -4.1e-05] | 0.180 | 3.2e-04 | 3.5e-05 | 2.4e-04 | 3.6e-05 |
| 8 | 1128857 | 0.152 | -2.4e-04 [-7.9e-04, -5.4e-05] | 0.152 | 2.4e-04 | 3.6e-05 | 1.9e-04 | 3.7e-05 |
| 9 | 1321937 | 0.128 | -2.6e-04 [-1.0e-03, -6.7e-05] | 0.128 | 1.9e-04 | 3.7e-05 | 1.5e-04 | 3.8e-05 |
| 10 | 1445537 | 0.107 | -3.0e-04 [-1.4e-03, -8.2e-05] | 0.107 | 1.5e-04 | 3.8e-05 | 1.2e-04 | 3.9e-05 |
| 11 | 1485123 | 0.087 | -3.9e-04 [-1.9e-03, -1.0e-04] | 0.087 | 1.2e-04 | 3.9e-05 | 1.0e-04 | 4.0e-05 |
| 12 | 1443856 | 0.069 | -6.1e-04 [-2.3e-03, -1.3e-04] | 0.069 | 1.0e-04 | 4.0e-05 | 8.6e-05 | 4.0e-05 |
| 15 | 1049822 | 0.030 | -1.6e-03 [-3.3e-03, -4.0e-04] | 0.030 | 6.4e-05 | 4.2e-05 | 5.6e-05 | 4.2e-05 |
| 20 | 261758 | 0.006 | -2.3e-03 [-3.8e-03, -1.4e-03] | 0.006 | 3.5e-05 | 4.4e-05 | 3.2e-05 | 4.5e-05 |
| 25 | 18884 | 0.002 | -2.6e-03 [-4.0e-03, -1.7e-03] | 0.002 | 2.2e-05 | 4.6e-05 | 2.1e-05 | 4.7e-05 |

## Does acceptance fall with K, and are the binding gains at large K centred near zero?

**Acceptance falls monotonically with K** in every family and at both search probabilities. At α = 0.3:

| K | acceptance |
|---|---|
| 2 | 0.92–0.98 |
| 3 | 0.55–0.64 |
| 4 | 0.38–0.42 |
| 5 | 0.28–0.31 |
| 8 | 0.13–0.17 |
| 12 | 0.04–0.08 |
| ≥ 20 | ≈ 0 |

The profile is almost the same at p = 0.05 and p = 1. Search intensity changes how many large proposals arise,
not how they are judged.

**Over most of the observed range, the rejections are not negative in expectation.**

- **Mean margin.** By the Section 2 formula, the marginal gain of the K-th member exceeds the marginal cost up to
  K ≈ 12 (normal), 16 (laplace) and 17 (t₃). At K = 4–8 the gain is 3–30 times the cost: for normal, ΔG = 4.6e-4
  against ΔΦ = 3.1e-5 at K = 4, and ΔG = 9.9e-5 against ΔΦ = 3.6e-5 at K = 8 (the "about 1e-4 for the eighth
  member" of the brief).
- **Acceptance.** Yet acceptance at these K is already 0.13–0.42.
- **Binding gain.** Its median is just below zero at K = 4–8 (−4e-5 to −3e-4). Its interquartile range is
  2–14e-4 wide. It straddles zero at K = 4–5 and ends within 6e-5 of zero at K = 6–8, with a share > 0 of
  0.13–0.42.
  - These margins are of the same order as ΔG itself, and an order of magnitude below the sampling noise of a
    50-period window. For a member's demeaned return, that noise is about σ/√50 ≈ 0.02–0.03 per period in level.
    It is smaller in the gain difference, but still well above 1e-4.

**So at K = 4–8, consent fails because the expected marginal gain is small next to the noise in each member's
estimate, and unanimity takes the minimum over K such estimates.** The minimum of K noisy draws of a small
positive number is negative most of the time, and more often the larger K is.

**From K ≈ 11–12 on, the binding gains move clearly below zero.**

| K | median binding gain (normal → t₃) | interquartile range |
|---|---|---|
| 12 | −3e-4 to −1e-3 | entirely below zero |
| 20 | −1e-3 to −2.3e-3 | — |

Here two things add up:
- the minimum over more members is lower;
- past the analytical crossing (K ≈ 12–17), the marginal member's gain is negative in expectation, because
  ΔΦ > ΔG.

The binding gains at large K are therefore not centred near zero. They are centred near zero only in the range
K = 3–8, where most proposals are made and where scope actually forms (K ≈ 4). In that range, consent rather
than cost is what binds.

**Under heavier tails** the binding gains spread more. The median sits further below zero under t₃ than under
normal at every K ≥ 4, and under laplace in most cells. The expected gain is larger there, but so is the estimation noise.
