# C36 note: the moment rule

Branch `referee-fixes`, from C35 (`252760d`).

Commits:
- **C36a:** `decision_rule='moment'` (code 2) in the merger kernel, the exit test and `find_optimal_alpha`, with
  the new `member_gain_moment`, the tests in `tests/test_moment_rule.py`, and the identity check;
- **C36b:** the `moment` design block (255 runs, scenario ids 1890–2144), the runs, the outputs and this note.

Runs were on Festus in `/scratch/bt307958/c36/IOxEE`. The tree is copied to `/groups/m-larch/bt307958/IOxEE_c36`
and verified by checksum. Departures from the brief are listed in the C36 section of `deviations.md`. The rule is
described in `MODEL_FLAGS.md`, section Decision Rule.

## 0. The rule, and one departure from the brief

The members' demeaned log returns over the 50-period window are u_jτ = ε_jτ − g. With sharing weights c_jτ
(proportional: s_jτ/Σs; equal: s_jτ/(K s_iτ)):
- y_iτ = (1 − α)u_iτ + α Σ_j c_jτ u_jτ;
- Q_iτ = (1 − α)u_iτ² + α Σ_j c_jτ u_jτ²;
- A_τ = e^g(1 − α + α c_iτ) + α(1 − c_iτ) − Φ c_iτ, and r_τ = e^g / A_τ.

Then

  Δ̃_i = (1/h) Σ_τ [log A_τ − g] + 1/(2(h − 1)) Σ_τ [r_τ Q_iτ − r_τ² (y_iτ − ȳ_i)²].

Δ̃_i = −∞ if h < 3, if any A_τ ≤ 0, or if any period's replayed p ≤ −1 (as under loggain). Δ̃ replaces Δ̂ wherever
loggain uses it: the merger test, the exit test and α renegotiation.

> **Departure: Δ̃ contains the arithmetic-mean transfer term, which the brief's formula leaves out.**
> - With equal member variances and Φ = 0, Δ̃ reduces to the brief's ½[V̂(ε_i) − V̂(y_i)]. With unequal variances,
>   the second-order expansion of loggain's replayed log gain also contains ½α(Σ_j ω_j σ_j² − σ_i²). This is the
>   part of the high-volatility members' arithmetic-mean premium that the pool passes to low-volatility members.
> - Market IQRs here span 0.1–0.3, so variances differ by up to a factor of 9. Without the term, the moment rule
>   would change the objective, not only its precision, and "moment − loggain" would mix the two.
> - On a 200,000-period window with heterogeneous members, Δ̃ matches loggain's Δ̂ within 10% under both sharing
>   rules. The brief's form misses the low-volatility member's gain by more than half
>   (`test_heterogeneous_members_match_loggain_on_long_window`).
> - So the moment − loggain gap below measures precision alone: same objective, better estimator.
> - Switching to the brief's literal form is a few lines in `member_gain_moment` and one rerun of the block.

## 1. Main block: K\*, K under moment, K under loggain (power law, p = 0.05)

- K is the median over 5 reps of the run's post-burn-in mean unit size.
- K\* is from `analytics/benchmarks.csv` (M = 50, IQR 0.2), which tabulates α ∈ {0.1, 0.3, 0.5}.
- Each moment run uses the same seed as its loggain counterpart.
- The other columns are medians over reps, shown as moment / loggain.
- Floor exits and voluntary exits are per period.

| log_family | alpha | K* | K mom | K log | K/K* mom | K/K* log | pooled share mom / log | Hill mom / log | mergers/period mom / log | floor exits mom / log | voluntary exits mom / log | acceptance mom / log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laplace | 0 | — | — | — | — | — | 0.000 / 0.000 | 1.064 / 1.064 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.000 / 0.000 |
| laplace | 0.01 | — | 2.462 | 2.287 | — | — | 0.557 / 0.365 | 1.066 / 1.067 | 48.52 / 46.89 | 47.83 / 44.43 | 0.66 / 2.44 | 0.399 / 0.381 |
| laplace | 0.02 | — | 2.752 | 2.469 | — | — | 0.675 / 0.459 | 1.075 / 1.069 | 49.84 / 49.65 | 49.52 / 47.25 | 0.30 / 2.39 | 0.416 / 0.408 |
| laplace | 0.05 | — | 3.223 | 2.747 | — | — | 0.783 / 0.563 | 1.092 / 1.075 | 49.51 / 50.20 | 49.40 / 48.15 | 0.09 / 2.02 | 0.427 / 0.419 |
| laplace | 0.1 | 10 | 3.708 | 3.005 | 0.371 | 0.300 | 0.853 / 0.634 | 1.105 / 1.098 | 47.44 / 48.57 | 47.39 / 46.77 | 0.02 / 1.78 | 0.428 / 0.413 |
| laplace | 0.2 | — | 4.500 | 3.458 | — | — | 0.899 / 0.703 | 1.138 / 1.113 | 42.43 / 43.70 | 42.40 / 42.33 | 0.00 / 1.34 | 0.417 / 0.389 |
| laplace | 0.3 | 16 | 5.401 | 3.999 | 0.338 | 0.250 | 0.914 / 0.782 | 1.195 / 1.165 | 36.29 / 37.76 | 36.26 / 36.88 | 0.00 / 0.85 | 0.402 / 0.361 |
| laplace | 0.4 | — | 6.471 | 4.798 | — | — | 0.924 / 0.857 | 1.256 / 1.220 | 30.22 / 31.80 | 30.20 / 31.30 | 0.00 / 0.46 | 0.391 / 0.339 |
| laplace | 0.5 | 20 | 7.760 | 5.948 | 0.388 | 0.297 | 0.934 / 0.906 | 1.344 / 1.329 | 24.55 / 25.24 | 24.49 / 25.05 | 0.00 / 0.18 | 0.379 / 0.321 |
| normal | 0 | — | — | — | — | — | 0.000 / 0.000 | 1.073 / 1.073 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.000 / 0.000 |
| normal | 0.01 | — | 2.304 | 2.156 | — | — | 0.420 / 0.216 | 1.074 / 1.073 | 35.03 / 31.87 | 33.54 / 28.67 | 1.50 / 3.18 | 0.287 / 0.258 |
| normal | 0.02 | — | 2.618 | 2.360 | — | — | 0.590 / 0.333 | 1.078 / 1.073 | 36.42 / 38.09 | 35.80 / 34.13 | 0.64 / 3.98 | 0.304 / 0.312 |
| normal | 0.05 | — | 3.153 | 2.705 | — | — | 0.763 / 0.473 | 1.096 / 1.081 | 36.31 / 39.54 | 36.13 / 36.03 | 0.19 / 3.56 | 0.316 / 0.331 |
| normal | 0.1 | 8 | 3.683 | 3.040 | 0.460 | 0.380 | 0.850 / 0.561 | 1.113 / 1.087 | 34.15 / 37.70 | 34.10 / 34.76 | 0.06 / 2.99 | 0.313 / 0.325 |
| normal | 0.2 | — | 4.557 | 3.600 | — | — | 0.915 / 0.651 | 1.160 / 1.130 | 29.77 / 32.93 | 29.75 / 30.81 | 0.01 / 2.18 | 0.303 / 0.302 |
| normal | 0.3 | 12 | 5.478 | 4.288 | 0.456 | 0.357 | 0.939 / 0.733 | 1.209 / 1.169 | 24.99 / 27.32 | 24.99 / 26.03 | 0.00 / 1.37 | 0.290 / 0.276 |
| normal | 0.4 | — | 6.469 | 5.229 | — | — | 0.948 / 0.828 | 1.268 / 1.235 | 20.18 / 21.47 | 20.18 / 20.84 | 0.00 / 0.70 | 0.280 / 0.252 |
| normal | 0.5 | 13 | 7.701 | 6.337 | 0.592 | 0.487 | 0.956 / 0.912 | 1.342 / 1.316 | 15.57 / 16.29 | 15.59 / 16.06 | 0.00 / 0.24 | 0.269 / 0.235 |
| t3 | 0 | — | — | — | — | — | 0.000 / 0.000 | 0.999 / 0.999 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.000 / 0.000 |
| t3 | 0.01 | — | 2.451 | 2.286 | — | — | 0.555 / 0.388 | 1.003 / 0.997 | 53.68 / 51.31 | 53.04 / 49.23 | 0.64 / 2.10 | 0.440 / 0.418 |
| t3 | 0.02 | — | 2.695 | 2.442 | — | — | 0.657 / 0.461 | 1.004 / 1.000 | 55.19 / 54.25 | 54.88 / 52.28 | 0.32 / 2.02 | 0.459 / 0.446 |
| t3 | 0.05 | — | 3.096 | 2.665 | — | — | 0.764 / 0.563 | 1.012 / 1.005 | 55.11 / 55.05 | 55.00 / 53.33 | 0.10 / 1.74 | 0.470 / 0.457 |
| t3 | 0.1 | 20 | 3.475 | 2.866 | 0.174 | 0.143 | 0.827 / 0.627 | 1.035 / 1.018 | 53.09 / 53.69 | 53.06 / 52.13 | 0.04 / 1.59 | 0.469 / 0.452 |
| t3 | 0.2 | — | 4.121 | 3.217 | — | — | 0.875 / 0.703 | 1.065 / 1.052 | 48.14 / 49.39 | 48.13 / 48.15 | 0.00 / 1.28 | 0.453 / 0.429 |
| t3 | 0.3 | 47 | 4.840 | 3.667 | 0.103 | 0.078 | 0.896 / 0.773 | 1.116 / 1.096 | 42.50 / 44.12 | 42.49 / 43.18 | 0.00 / 0.93 | 0.435 / 0.403 |
| t3 | 0.4 | — | 5.685 | 4.277 | — | — | 0.908 / 0.834 | 1.164 / 1.140 | 36.69 / 38.49 | 36.68 / 37.97 | 0.00 / 0.56 | 0.420 / 0.379 |
| t3 | 0.5 | 50 | 6.777 | 5.063 | 0.136 | 0.101 | 0.920 / 0.884 | 1.221 / 1.182 | 31.17 / 32.60 | 31.15 / 32.28 | 0.00 / 0.31 | 0.409 / 0.359 |

**K under moment minus K under loggain (the verifiability gap), main block, p = 0.05**

| family | α = 0.1 | α = 0.3 | α = 0.5 | pooled share gap, α = 0.1 / 0.3 / 0.5 |
|---|---|---|---|---|
| normal | +0.64 (+21%) | +1.19 (+28%) | +1.36 (+22%) | +0.29 / +0.21 / +0.04 |
| laplace | +0.70 (+23%) | +1.40 (+35%) | +1.81 (+30%) | +0.22 / +0.13 / +0.03 |
| t₃ | +0.61 (+21%) | +1.17 (+32%) | +1.71 (+34%) | +0.20 / +0.12 / +0.04 |

## 2. Cost level (power law, c0 × 5 and × 20, laplace and t₃, p = 0.05)

The ×1 rows are in the main table above. The loggain values are the C35 cost-level-high cells.

| log_family | cost_multiplier | alpha | K* | K mom | K log | K/K* mom | K/K* log | pooled share mom / log | Hill mom / log | mergers/period mom / log | floor exits mom / log | voluntary exits mom / log | acceptance mom / log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laplace | 5 | 0.1 | 5 | 2.793 | 2.480 | 0.559 | 0.496 | 0.677 / 0.438 | 1.084 / 1.079 | 42.65 / 44.07 | 42.27 / 41.09 | 0.38 / 2.99 | 0.359 / 0.363 |
| laplace | 5 | 0.3 | 8 | 3.883 | 3.138 | 0.485 | 0.392 | 0.871 / 0.636 | 1.172 / 1.137 | 32.73 / 36.09 | 32.70 / 34.18 | 0.04 / 1.90 | 0.306 / 0.314 |
| laplace | 5 | 0.5 | 9 | 5.273 | 4.129 | 0.586 | 0.459 | 0.931 / 0.815 | 1.289 / 1.254 | 22.23 / 24.55 | 22.22 / 23.97 | 0.00 / 0.63 | 0.253 / 0.244 |
| laplace | 20 | 0.1 | 3 | 2.206 | 2.120 | 0.735 | 0.707 | 0.354 / 0.208 | 1.064 / 1.057 | 38.24 / 31.87 | 36.48 / 29.64 | 1.76 / 2.07 | 0.310 / 0.257 |
| laplace | 20 | 0.3 | 4 | 2.810 | 2.452 | 0.702 | 0.613 | 0.667 / 0.410 | 1.130 / 1.079 | 34.03 / 37.31 | 33.50 / 33.80 | 0.52 / 3.51 | 0.287 / 0.306 |
| laplace | 20 | 0.5 | 5 | 3.480 | 2.847 | 0.696 | 0.569 | 0.818 / 0.566 | 1.233 / 1.131 | 24.98 / 29.64 | 24.85 / 27.22 | 0.14 / 2.43 | 0.225 / 0.251 |
| t3 | 5 | 0.1 | 7 | 2.779 | 2.491 | 0.397 | 0.356 | 0.683 / 0.496 | 1.016 / 1.005 | 51.77 / 51.84 | 51.49 / 49.71 | 0.29 / 2.14 | 0.434 / 0.425 |
| t3 | 5 | 0.3 | 10 | 3.727 | 3.042 | 0.373 | 0.304 | 0.855 / 0.662 | 1.096 / 1.065 | 41.93 / 44.13 | 41.91 / 42.69 | 0.04 / 1.45 | 0.383 / 0.379 |
| t3 | 5 | 0.5 | 13 | 4.975 | 3.881 | 0.383 | 0.299 | 0.908 / 0.814 | 1.205 / 1.172 | 30.91 / 33.80 | 30.90 / 33.25 | 0.00 / 0.55 | 0.332 / 0.318 |
| t3 | 20 | 0.1 | 3 | 2.278 | 2.161 | 0.759 | 0.720 | 0.440 / 0.294 | 1.006 / 0.997 | 47.37 / 42.12 | 46.04 / 40.00 | 1.33 / 2.09 | 0.386 / 0.341 |
| t3 | 20 | 0.3 | 5 | 2.841 | 2.496 | 0.568 | 0.499 | 0.699 / 0.477 | 1.048 / 1.014 | 42.63 / 44.61 | 42.22 / 42.05 | 0.41 / 2.57 | 0.360 / 0.368 |
| t3 | 20 | 0.5 | 6 | 3.519 | 2.884 | 0.586 | 0.481 | 0.827 / 0.631 | 1.144 / 1.068 | 33.21 / 36.81 | 33.09 / 35.16 | 0.12 / 1.66 | 0.298 / 0.313 |

**K under moment minus K under loggain, by cost level (α = 0.1 / 0.3 / 0.5)**

| family | ×1 | ×5 | ×20 |
|---|---|---|---|
| laplace | +0.70 / +1.40 / +1.81 | +0.31 / +0.75 / +1.14 | +0.09 / +0.36 / +0.63 |
| t₃ | +0.61 / +1.17 / +1.71 | +0.29 / +0.69 / +1.09 | +0.12 / +0.35 / +0.64 |

## 3. Search p = 1 (power law, α ∈ {0.1, 0.3, 0.5})

| log_family | alpha | K* | K mom | K log | K/K* mom | K/K* log | pooled share mom / log | Hill mom / log | mergers/period mom / log | floor exits mom / log | voluntary exits mom / log | acceptance mom / log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laplace | 0.1 | 10 | 5.319 | 4.634 | 0.532 | 0.463 | 0.967 / 0.885 | 1.100 / 1.088 | 164.13 / 166.25 | 164.09 / 164.80 | 0.02 / 1.43 | 0.107 / 0.097 |
| laplace | 0.3 | 16 | 8.049 | 7.144 | 0.503 | 0.446 | 0.983 / 0.944 | 1.179 / 1.169 | 116.15 / 119.03 | 116.12 / 118.66 | 0.00 / 0.36 | 0.136 / 0.119 |
| laplace | 0.5 | 20 | 11.110 | 10.656 | 0.555 | 0.533 | 0.988 / 0.987 | 1.314 / 1.307 | 71.71 / 73.22 | 71.70 / 73.19 | 0.00 / 0.02 | 0.167 / 0.157 |
| normal | 0.1 | 8 | 5.031 | 4.511 | 0.629 | 0.564 | 0.957 / 0.856 | 1.118 / 1.111 | 113.96 / 121.39 | 113.92 / 118.76 | 0.04 / 2.59 | 0.070 / 0.070 |
| normal | 0.3 | 12 | 7.410 | 7.152 | 0.618 | 0.596 | 0.986 / 0.925 | 1.200 / 1.182 | 77.91 / 83.27 | 77.91 / 82.69 | 0.00 / 0.59 | 0.082 / 0.085 |
| normal | 0.5 | 13 | 10.013 | 10.451 | 0.770 | 0.804 | 0.991 / 0.984 | 1.325 / 1.322 | 46.38 / 46.56 | 46.38 / 46.53 | 0.00 / 0.03 | 0.100 / 0.111 |
| t3 | 0.1 | 20 | 5.360 | 4.500 | 0.268 | 0.225 | 0.963 / 0.886 | 1.019 / 1.023 | 192.62 / 194.35 | 192.58 / 193.24 | 0.02 / 1.12 | 0.127 / 0.111 |
| t3 | 0.3 | 47 | 7.937 | 6.661 | 0.169 | 0.142 | 0.979 / 0.954 | 1.104 / 1.104 | 140.58 / 144.23 | 140.56 / 143.91 | 0.00 / 0.35 | 0.155 / 0.130 |
| t3 | 0.5 | 50 | 10.832 | 9.775 | 0.217 | 0.195 | 0.985 / 0.984 | 1.240 / 1.229 | 92.61 / 96.94 | 92.59 / 96.86 | 0.00 / 0.07 | 0.175 / 0.162 |

**K under moment minus K under loggain, p = 1:**

| family | α = 0.1 | α = 0.3 | α = 0.5 |
|---|---|---|---|
| normal | +0.52 | +0.26 | −0.44 |
| laplace | +0.69 | +0.91 | +0.45 |
| t₃ | +0.86 | +1.28 | +1.06 |

## 4. Acceptance by K, moment against loggain (α = 0.3)

- **Acceptance** = post-burn-in mergers / proposals at merged size K, pooled over 5 reps.
- **Binding gain** = min_i [gain_i(Ĉ) − max(0, gain_i(C_i))], median, where gain is Δ̃ under moment and Δ̂ under
  loggain.
- **ΔG into K** is the Section 2 marginal gain of the K-th member, ½α(2 − α)σ²[1/(K−1) − 1/K], at the median
  market σ (normal). **ΔΦ into K** = Φ(K) − Φ(K−1).
- Full columns (p25/p75, the K → K+1 margins, all families) are in `pilot_c/acceptance_by_K_moment.csv`
  (per run) and `pilot_c/acceptance_by_K_moment_pooled.csv`. The loggain columns repeat C35's
  `pilot_c/acceptance_by_K*.csv`.

**p = 0.05**

| K | normal acc. mom / log | normal proposals mom / log | normal binding median mom / log | laplace acc. mom / log | laplace proposals mom / log | laplace binding median mom / log | t3 acc. mom / log | t3 proposals mom / log | t3 binding median mom / log | ΔG into K (normal) | ΔΦ into K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.999 / 0.919 | 124340 / 172680 | 2.7e-03 / 1.6e-03 | 1.000 / 0.965 | 195942 / 246897 | 5.0e-03 / 3.5e-03 | 1.000 / 0.970 | 251505 / 308316 | 5.7e-03 / 4.0e-03 | 2.8e-03 | 2.6e-05 |
| 3 | 0.897 / 0.553 | 45301 / 111008 | 5.6e-04 / 5.6e-05 | 0.870 / 0.593 | 70438 / 148739 | 9.8e-04 / 1.7e-04 | 0.760 / 0.569 | 104961 / 195833 | 8.3e-04 / 1.5e-04 | 9.3e-04 | 2.9e-05 |
| 4 | 0.677 / 0.375 | 41816 / 113850 | 1.1e-04 / -7.0e-05 | 0.680 / 0.409 | 57398 / 141933 | 2.1e-04 / -8.3e-05 | 0.589 / 0.393 | 85605 / 180853 | 1.2e-04 / -1.3e-04 | 4.6e-04 | 3.1e-05 |
| 5 | 0.475 / 0.275 | 60724 / 127760 | -7.4e-06 / -9.7e-05 | 0.536 / 0.313 | 66919 / 146316 | 2.5e-05 / -1.4e-04 | 0.482 / 0.300 | 91627 / 174660 | -1.3e-05 / -2.0e-04 | 2.8e-04 | 3.3e-05 |
| 6 | 0.349 / 0.210 | 94361 / 140947 | -3.3e-05 / -1.1e-04 | 0.430 / 0.248 | 87616 / 149277 | -2.7e-05 / -1.6e-04 | 0.396 / 0.242 | 106877 / 165257 | -5.2e-05 / -2.2e-04 | 1.9e-04 | 3.4e-05 |
| 7 | 0.265 / 0.165 | 128055 / 148752 | -4.4e-05 / -1.2e-04 | 0.354 / 0.202 | 109665 / 145749 | -4.5e-05 / -1.8e-04 | 0.336 / 0.196 | 121072 / 148813 | -6.9e-05 / -2.6e-04 | 1.3e-04 | 3.5e-05 |
| 8 | 0.205 / 0.129 | 143487 / 146505 | -5.4e-05 / -1.5e-04 | 0.295 / 0.169 | 127387 / 135349 | -5.6e-05 / -1.9e-04 | 0.282 / 0.163 | 129089 / 125889 | -8.2e-05 / -3.1e-04 | 9.9e-05 | 3.6e-05 |
| 9 | 0.158 / 0.099 | 138338 / 132093 | -6.5e-05 / -1.9e-04 | 0.244 / 0.137 | 130287 / 118288 | -6.6e-05 / -2.4e-04 | 0.241 / 0.135 | 125266 / 102679 | -9.2e-05 / -3.9e-04 | 7.7e-05 | 3.7e-05 |
| 10 | 0.115 / 0.073 | 122688 / 110618 | -8.2e-05 / -2.9e-04 | 0.199 / 0.109 | 120652 / 97593 | -7.8e-05 / -3.3e-04 | 0.201 / 0.114 | 111661 / 78745 | -1.1e-04 / -5.4e-04 | 6.2e-05 | 3.8e-05 |
| 11 | 0.073 / 0.051 | 101868 / 87478 | -1.4e-04 / -4.3e-04 | 0.161 / 0.087 | 103238 / 74379 | -9.5e-05 / -5.2e-04 | 0.163 / 0.091 | 93946 / 58207 | -1.5e-04 / -7.4e-04 | 5.1e-05 | 3.9e-05 |
| 12 | 0.043 / 0.038 | 81003 / 65707 | -2.9e-04 / -5.6e-04 | 0.121 / 0.069 | 82807 / 55435 | -1.4e-04 / -7.2e-04 | 0.130 / 0.072 | 73896 / 39696 | -2.0e-04 / -1.0e-03 | 4.2e-05 | 4.0e-05 |
| 15 | 0.004 / 0.009 | 36926 / 20634 | -6.6e-04 / -8.4e-04 | 0.038 / 0.029 | 33922 / 16717 | -7.7e-04 / -1.2e-03 | 0.053 / 0.032 | 27468 / 9744 | -9.4e-04 / -1.6e-03 | 2.6e-05 | 4.2e-05 |
| 20 | 0.000 / 0.000 | 2077 / 1041 | -8.0e-04 / -1.0e-03 | 0.002 / 0.004 | 3488 / 935 | -1.3e-03 / -1.7e-03 | 0.005 / 0.000 | 2392 / 376 | -1.9e-03 / -2.1e-03 | 1.5e-05 | 4.4e-05 |
| 25 | 0.000 / 0.000 | 7 / 10 | -7.7e-04 / -1.3e-03 | 0.000 / 0.000 | 62 / 15 | -1.3e-03 / -1.6e-03 | 0.000 / 0.000 | 54 / 5 | -2.0e-03 / -2.2e-03 | 9.5e-06 | 4.6e-05 |
| 30 | — / — | 0 / 0 | — / — | 0.000 / — | 1 / 0 | -9.5e-04 / — | — / — | 0 / 0 | — / — | — | — |

**p = 1**

| K | normal acc. mom / log | normal proposals mom / log | normal binding median mom / log | laplace acc. mom / log | laplace proposals mom / log | laplace binding median mom / log | t3 acc. mom / log | t3 proposals mom / log | t3 binding median mom / log | ΔG into K (normal) | ΔΦ into K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.999 / 0.925 | 245130 / 259919 | 2.7e-03 / 1.8e-03 | 1.000 / 0.971 | 355058 / 373897 | 4.4e-03 / 3.2e-03 | 1.000 / 0.981 | 487120 / 505018 | 5.3e-03 / 3.9e-03 | 2.9e-03 | 2.6e-05 |
| 3 | 0.940 / 0.608 | 91458 / 145048 | 7.2e-04 / 1.5e-04 | 0.886 / 0.638 | 139217 / 210133 | 1.1e-03 / 3.2e-04 | 0.758 / 0.595 | 229424 / 321070 | 1.0e-03 / 2.8e-04 | 9.6e-04 | 2.9e-05 |
| 4 | 0.735 / 0.404 | 71252 / 157770 | 2.4e-04 / -4.0e-05 | 0.702 / 0.421 | 102232 / 222304 | 3.7e-04 / -6.0e-05 | 0.581 / 0.387 | 175676 / 347071 | 2.1e-04 / -1.3e-04 | 4.8e-04 | 3.1e-05 |
| 5 | 0.428 / 0.279 | 121034 / 240195 | -1.8e-05 / -7.4e-05 | 0.501 / 0.302 | 125680 / 306466 | 9.9e-07 / -1.2e-04 | 0.429 / 0.278 | 198834 / 471513 | -8.4e-05 / -2.2e-04 | 2.9e-04 | 3.3e-05 |
| 6 | 0.259 / 0.210 | 283805 / 399410 | -4.5e-05 / -8.1e-05 | 0.348 / 0.236 | 222465 / 460472 | -5.5e-05 / -1.4e-04 | 0.319 / 0.219 | 290407 / 661457 | -1.5e-04 / -2.3e-04 | 1.9e-04 | 3.4e-05 |
| 7 | 0.188 / 0.167 | 582178 / 633911 | -5.2e-05 / -8.3e-05 | 0.263 / 0.193 | 414114 / 665807 | -6.7e-05 / -1.4e-04 | 0.253 / 0.180 | 453798 / 893428 | -1.5e-04 / -2.3e-04 | 1.4e-04 | 3.5e-05 |
| 8 | 0.147 / 0.135 | 920647 / 888171 | -5.9e-05 / -8.9e-05 | 0.212 / 0.163 | 681278 / 892081 | -7.4e-05 / -1.4e-04 | 0.210 / 0.152 | 676272 / 1128857 | -1.5e-04 / -2.4e-04 | 1.0e-04 | 3.6e-05 |
| 9 | 0.117 / 0.109 | 1158951 / 1100110 | -6.6e-05 / -9.7e-05 | 0.179 / 0.138 | 941590 / 1095911 | -8.1e-05 / -1.5e-04 | 0.180 / 0.128 | 921420 / 1321937 | -1.5e-04 / -2.6e-04 | 8.0e-05 | 3.7e-05 |
| 10 | 0.089 / 0.087 | 1230232 / 1232093 | -7.8e-05 / -1.3e-04 | 0.151 / 0.116 | 1130945 / 1238703 | -9.1e-05 / -1.7e-04 | 0.154 / 0.107 | 1116674 / 1445537 | -1.6e-04 / -3.0e-04 | 6.4e-05 | 3.8e-05 |
| 11 | 0.063 / 0.066 | 1195198 / 1295349 | -1.1e-04 / -2.0e-04 | 0.126 / 0.095 | 1200801 / 1303235 | -1.1e-04 / -2.2e-04 | 0.131 / 0.087 | 1222024 / 1485123 | -1.7e-04 / -3.9e-04 | 5.2e-05 | 3.9e-05 |
| 12 | 0.037 / 0.047 | 1164386 / 1295642 | -1.9e-04 / -4.2e-04 | 0.099 / 0.075 | 1165540 / 1303218 | -1.4e-04 / -3.4e-04 | 0.108 / 0.069 | 1232021 / 1443856 | -1.9e-04 / -6.1e-04 | 4.4e-05 | 4.0e-05 |
| 15 | 0.002 / 0.012 | 1203756 / 1081880 | -6.6e-04 / -8.7e-04 | 0.029 / 0.030 | 886067 / 1024002 | -6.1e-04 / -1.2e-03 | 0.048 / 0.030 | 925935 / 1049822 | -5.0e-04 / -1.6e-03 | 2.7e-05 | 4.2e-05 |
| 20 | 0.000 / 0.001 | 263746 / 283658 | -8.0e-04 / -1.1e-03 | 0.001 / 0.005 | 349206 / 297245 | -1.3e-03 / -1.7e-03 | 0.006 / 0.006 | 330228 / 261758 | -2.0e-03 / -2.3e-03 | 1.5e-05 | 4.4e-05 |
| 25 | 0.000 / 0.000 | 3762 / 13832 | -9.1e-04 / -1.3e-03 | 0.000 / 0.001 | 22282 / 22059 | -1.5e-03 / -1.9e-03 | 0.002 / 0.002 | 33997 / 18884 | -2.4e-03 / -2.6e-03 | 9.6e-06 | 4.6e-05 |
| 30 | 0.000 / 0.000 | 2 / 157 | -8.5e-04 / -1.4e-03 | 0.000 / 0.000 | 157 / 419 | -1.6e-03 / -2.0e-03 | 0.001 / 0.003 | 699 / 333 | -2.7e-03 / -2.8e-03 | 6.6e-06 | 4.8e-05 |

## 5. Adopted α (endogenous α, power law, start α = 0.1, p = 0.05)

The moment rows are the 15 moment-block runs. The loggain rows are the 15 loggain endogenous-α power-law runs with
the same seeds.
- **Run-level columns:** from `tidy.csv`, medians over 5 runs of the run-level adopted α.
- **Unit-level columns:** from `pilot_c/alpha_scatter_moment.csv` and `pilot_c/alpha_scatter.csv`. They cover units
  alive at T; α_tavg is the unit's time-averaged α, and mean IQR is the mean IQR of its members' markets.

| family | α median (run medians) mom / log | α mean mom / log | units at T mom / log | α_tavg mean mom / log | share α_tavg ≥ 0.95 mom / log | Spearman ρ(α_tavg, mean IQR) mom / log | K mom / log |
|---|---|---|---|---|---|---|---|
| normal | 0.900 / 0.850 | 0.934 / 0.818 | 681 / 957 | 0.928 / 0.806 | 0.316 / 0.052 | 0.033 / 0.285 | 16.928 / 12.069 |
| laplace | 0.900 / 0.800 | 0.923 / 0.805 | 614 / 940 | 0.922 / 0.795 | 0.319 / 0.066 | -0.027 / 0.201 | 18.000 / 11.495 |
| t3 | 0.900 / 0.800 | 0.913 / 0.782 | 740 / 1048 | 0.902 / 0.780 | 0.315 / 0.069 | 0.018 / 0.156 | 14.279 / 8.758 |

## 6. σ̂²/σ² over 50-period windows

The sample variance (divisor h − 1) of a 50-period window, divided by the true variance:
- **i.i.d. rows:** 200,000 windows of draws at IQR 0.2, with σ² exact.
- **Trace rows:** windows of standalone firms with no floor hit in the α = 0.3 rep-0 trace (`traces_c29`), divided
  by the market's σ².
- The data are in `pilot_c/variance_ratio_c36.csv`.

| family | source | windows | median | q05 | q25 | q75 | q95 | share < ½ | share > 2 |
|---|---|---|---|---|---|---|---|---|---|
| normal | i.i.d. | 200,000 | 0.987 | 0.693 | 0.858 | 1.129 | 1.354 | 0.001 | 0.000 |
| laplace | i.i.d. | 200,000 | 0.955 | 0.566 | 0.774 | 1.178 | 1.580 | 0.022 | 0.008 |
| t₃ | i.i.d. | 200,000 | 0.774 | 0.418 | 0.591 | 1.056 | 1.997 | 0.129 | 0.050 |
| normal | trace | 5,911 | 1.013 | 0.269 | 0.806 | 1.206 | 1.503 | 0.108 | 0.002 |
| laplace | trace | 4,910 | 0.969 | 0.345 | 0.745 | 1.210 | 1.645 | 0.089 | 0.010 |
| t₃ | trace | 3,696 | 0.829 | 0.363 | 0.640 | 1.096 | 1.887 | 0.116 | 0.041 |

- Under t₃ the window variance is right-skewed. Its mean is 1.00, but its median is 0.77, so in more than half the
  windows the rule sees a variance below the true one. One window in eight sees less than half of it, and one in
  twenty sees more than twice it.
- The trace rows of normal and laplace have a lower tail the i.i.d. rows lack: 11% and 9% of windows are below ½.
  This comes from selecting windows with no floor hit, which drops windows that hold a large negative shock.

## 7. Identity report

- **Existing scenarios, at full T = 11,000.**
  - **Sample:** 148 reruns on the C36 code, compared with their C35 pickles by `analytics/identity_c35.py
    --same-schema`. It compares every field and every summary key as `pickle.dumps` bytes, with no key exempted, plus
    the tidy row as text.
  - The 148 are:
    - the C35 identity sample: one scenario of each of the 56 C34 cells outside the C35 reruns, plus the 90 main
      and search p = 1 reruns;
    - one run of each cost-level-high cell (1822, 1868).
    - So every block, both rules (replay and loggain), endogenous α and every cost level are covered.
  - **Result:** 148/148 identical. Counters are consistent in 148/148 (for the moment runs too, the binding-gain
    histogram holds one count per post-burn-in proposal). The median runtime ratio is 1.00. Details are in
    `diagnostics/identity_c36.csv`.
- **Outputs:**
  - `tidy.csv`: the header and the first 1,890 rows are byte-identical to C35, checked on Festus and against git
    HEAD.
  - `medians.csv`: 0 rows removed and 51 added (27 main, 12 cost-level, 9 search p = 1, 3 endogenous α).
  - `scenarios.json`: scenarios 0–1889 are unchanged, with 9,750 lines added and none removed.
  - `pilot_c/alpha_scatter.csv` is unchanged.
- **Tests:** `pytest tests/` on Festus: 236 passed, 0 failed.

## 8. Answers

**Does scope under the moment rule approach min(K\*, the attrition limit)? No.**
- **Distance to K\*.** At p = 0.05, K/K\* under moment is 0.34–0.59 for normal and laplace (α 0.1–0.5) and
  0.10–0.17 for t₃. Under loggain it is 0.25–0.49 and 0.08–0.14. The moment rule closes 3% (t₃, α = 0.3) to 20%
  (normal, α = 0.5) of the distance to K\*.
- **Consent still binds well before K\*, so scope is below the attrition limit too.** The pilot has no consent-free
  run, so that limit is not measured directly. But consent still rejects where the expected gain clearly exceeds
  the cost:
  - acceptance rises at every K ≤ 12. At K = 3 it goes from 0.55–0.59 to 0.76–0.90, and at K = 5 from 0.28–0.31 to
    0.48–0.54;
  - it still falls steeply with K: 0.20–0.30 at K = 8 and 0.04–0.13 at K = 12;
  - at K = 5, ΔG = 2.8e-4 is eight times ΔΦ = 3.3e-5, yet half the proposals are rejected. By Section 2 the
    marginal gain exceeds the marginal cost up to K ≈ 12–17;
  - the binding gain's median now crosses zero between K = 4 and 6 under moment, against between K = 3 and 4 under
    loggain.
  - Raising search to p = 1 lifts K under moment to 7.4–8.0 at α = 0.3, so at p = 0.05 the proposal rate also
    holds scope down.
- **Attrition is now almost entirely floor exits.** Voluntary exits fall to about 0 (from 0.85–1.37 per period at
  α = 0.3). Mergers per period roughly equal floor exits, as under loggain.
- **What still pulls the binding gain below ΔG.** At K = 2 the binding median under moment equals ΔG (normal:
  2.7e-3 against 2.8e-3). At K = 3 it is 0.6 ΔG and at K = 4 it is 0.24 ΔG. Two candidates remain, and these runs
  do not separate them:
  - unanimity takes the minimum of K estimates, each carrying σ̂² noise (Section 6);
  - the transfer term (Section 0). With market variances differing up to ninefold, a low-volatility entrant lowers
    the pool's average variance and so the transfer a high-volatility incumbent receives. That is a real loss to the
    incumbent, not noise.
  - The brief's literal form would remove the second one, at the price of a different objective.

**Does cost now bite? Yes, more than under loggain, but K still does not reach K\*.**
- **K responds more to cost.** At α = 0.3, K falls by 0.60 per doubling of c0 under moment for laplace (5.40, 3.88,
  2.81 at ×1, ×5, ×20), against 0.36 under loggain. For t₃ it is 0.46 (4.84, 3.73, 2.84) against 0.27. With less
  estimation noise, more acceptances turn on the expected gain against the cost.
- **The moment − loggain gap shrinks as cost rises:** laplace at α = 0.3 goes from +1.40 at ×1 to +0.75 at ×5 and
  +0.36 at ×20; t₃ goes from +1.17 to +0.69 and +0.35. The higher the cost, the more of what consent used to hold
  back the cost now holds back by itself.
- **K/K\* under moment:**

  | family | ×5 | ×20 |
  |---|---|---|
  | laplace | 0.49–0.59 | 0.70–0.74 |
  | t₃ | 0.37–0.40 | 0.57–0.76 |

- **K stays below K\* in every cell.** At ×20, K\* = 3–6 and K = 2.21–3.52. The shortfall is 0.7–2.5 members, and
  it is largest at α = 0.5 (t₃: 3.52 against 6). So even at ×20, cost is not the only constraint. It now binds
  alongside a consent constraint that is weaker than under loggain but still present.

**How large is the verifiability gap, and is it largest under t₃?**
- **Size.** At p = 0.05 the moment rule raises K by 0.6–0.7 at α = 0.1, 1.2–1.4 at α = 0.3 and 1.4–1.8 at α = 0.5,
  or 21–35%. It raises the pooled share by 0.20–0.29 at α = 0.1, with the effect fading as α rises (0.03–0.04 at
  α = 0.5).
- **It is not largest under t₃ at p = 0.05.** In K, laplace has the largest gap at every α (+0.70, +1.40, +1.81).
  t₃ is second (+0.61, +1.17, +1.71), and leads only in relative terms at α = 0.5 (34% against 30%). In pooled share
  and in acceptance at K = 3, normal gains most and t₃ least: acceptance rises by 0.34 (normal), 0.28 (laplace) and
  0.19 (t₃).
- **At p = 1 the gap is largest under t₃** at every α (+0.86, +1.28, +1.06, against +0.45 to +0.91 for laplace and
  −0.44 to +0.52 for normal).
- **Why t₃ gains less at p = 0.05.** The rule's own input is poorly measured under t₃.
  - Over a 50-period window, σ̂²/σ² under t₃ has median 0.77 and 5–95% range 0.42–2.00. One window in eight is
    below ½, and one in twenty above 2.
  - Under normal shocks the median is 0.99 (0.69–1.35).
  - Replacing the replayed log gain with second moments therefore helps least where the second moments themselves
    are pre-asymptotic. Under t₃, neither rule can verify the gain from 50 periods.
  - With K\* = 20–50, t₃ stays furthest from K\* under both rules (K/K\* 0.10–0.17 under moment).

**Adopted α.**
- Under the moment rule, units renegotiate to higher α. The median of run medians is 0.90 in all three families,
  against 0.80–0.85 under loggain.
- A third of surviving units sit at α_tavg ≥ 0.95, against 5–7%.
- Mean K in these runs is 14–18, against 9–12.
- The link between adopted α and the volatility of the members' markets disappears: Spearman ρ is −0.03 to 0.03,
  against 0.16–0.29 under loggain.
- Under loggain, the units that pooled most were those where the gain was large enough to show through the noise.
  Under moment, the gain is visible at every volatility level, so units pool close to fully everywhere.
