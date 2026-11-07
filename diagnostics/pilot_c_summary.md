# Phase C Pilot Summary

## Overview

- **Scenarios**: 2145
- **Total runtime**: 455330s (126.48 CPU-hours)
- **Mean ms/step**: 19.30 ± 14.88
- **Grid**: 50 × 50 (M × N); size block M, N ∈ {25, 100}

### Block counts

| Block | Count |
|-------|-------|
| correlation | 45 |
| cost-level | 360 |
| cost-level-full | 180 |
| cost-level-high | 90 |
| endogenous-alpha | 60 |
| equal-split | 180 |
| floor-level | 90 |
| lookback | 100 |
| main | 540 |
| moment | 255 |
| rule-replay | 45 |
| search | 120 |
| size | 80 |

## K versus K* by Family and Cost

Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `K_eff_post_burnin_median`
Reference: `analytics/benchmarks.csv` column `K_star`

### At α = 0.1

| Family | Cost | K [25,75] | K* | K_eff [25,75] | Acc. rate |
|--------|------|-----------|-----|---------------|-----------|
| laplace | exponential | 2.000 [2.000, 2.000] | 15 | 1.954 [1.953, 1.960] | 0.332 |
| laplace | linear | 3.000 [2.000, 3.000] | 10 | 1.960 [1.959, 1.964] | 0.402 |
| laplace | power_law | 3.000 [3.000, 3.000] | 10 | 1.964 [1.953, 1.970] | 0.413 |
| laplace | quadratic | 3.000 [3.000, 3.000] | 12 | 1.988 [1.985, 1.990] | 0.418 |
| normal | exponential | 2.000 [2.000, 2.000] | 12 | 1.949 [1.943, 1.957] | 0.182 |
| normal | linear | 3.000 [3.000, 3.000] | 7 | 1.988 [1.987, 1.988] | 0.303 |
| normal | power_law | 3.000 [3.000, 3.000] | 8 | 1.990 [1.990, 1.990] | 0.325 |
| normal | quadratic | 3.000 [3.000, 3.000] | 10 | 2.068 [2.050, 2.128] | 0.341 |
| t3 | exponential | 2.000 [2.000, 2.000] | 47 | 1.925 [1.922, 1.928] | 0.396 |
| t3 | linear | 2.000 [2.000, 2.000] | 28 | 1.919 [1.918, 1.926] | 0.479 |
| t3 | power_law | 2.000 [2.000, 2.000] | 20 | 1.934 [1.930, 1.937] | 0.452 |
| t3 | quadratic | 2.000 [2.000, 3.000] | 24 | 1.935 [1.935, 1.954] | 0.497 |

## Hill Exponent

Source: `pilot_c/tidy.csv` column `hill_exponent_median`

**Theoretical at α=0**: 1/(1−c) = 1/(1−0.1272) = 1.1457

### At α = 0

| Family | Hill median [25,75] |
|--------|---------------------|
| laplace | 1.064 [1.055, 1.073] |
| normal | 1.079 [1.071, 1.090] |
| t3 | 0.995 [0.986, 0.999] |

### Hill exponent versus α (laplace family)

| α | Hill median [25,75] |
|---|---------------------|
| 0.0 | 1.064 [1.055, 1.073] |
| 0.01 | 1.057 [1.054, 1.075] |
| 0.02 | 1.062 [1.054, 1.074] |
| 0.05 | 1.062 [1.060, 1.079] |
| 0.1 | 1.081 [1.071, 1.088] |
| 0.2 | 1.107 [1.097, 1.123] |
| 0.3 | 1.161 [1.148, 1.174] |
| 0.4 | 1.217 [1.208, 1.247] |
| 0.5 | 1.313 [1.298, 1.333] |

![Hill vs α](pilot_c_hill_vs_alpha.png)

## Floor-hit Rate by Status

Source: `pilot_c/tidy.csv` columns `floor_hit_rate_standalone`, `floor_hit_rate_member`

### Laplace family

| α | Standalone [25,75] | Member [25,75] |
|---|-------------------|----------------|
| 0.0 | 0.115 [0.109, 0.121] | 0.000 [0.000, 0.000] |
| 0.01 | 0.101 [0.096, 0.105] | 0.015 [0.009, 0.017] |
| 0.02 | 0.098 [0.091, 0.103] | 0.018 [0.013, 0.019] |
| 0.05 | 0.094 [0.088, 0.100] | 0.019 [0.016, 0.020] |
| 0.1 | 0.089 [0.084, 0.094] | 0.018 [0.017, 0.020] |
| 0.2 | 0.080 [0.076, 0.085] | 0.017 [0.016, 0.018] |
| 0.3 | 0.072 [0.067, 0.077] | 0.015 [0.014, 0.016] |
| 0.4 | 0.064 [0.059, 0.068] | 0.013 [0.012, 0.014] |
| 0.5 | 0.054 [0.051, 0.059] | 0.010 [0.009, 0.011] |

![Floor-hit rate](pilot_c_floor_hit.png)

## HHI and Top-10 Share

Source: `pilot_c/tidy.csv` columns `hhi_within_median`, `hhi_aggregate_median`, `top10pct_aggregate_median`

### Laplace family

| α | HHI within [25,75] | HHI agg [25,75] | Top-10 [25,75] | Cong. share [25,75] |
|---|-------------------|-----------------|----------------|---------------------|
| 0.0 | 0.181 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.656 [0.654, 0.661] | 0.000 [0.000, 0.000] |
| 0.01 | 0.181 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.651 [0.646, 0.654] | 0.232 [0.144, 0.294] |
| 0.02 | 0.180 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.654 [0.647, 0.658] | 0.362 [0.246, 0.421] |
| 0.05 | 0.179 [0.177, 0.182] | 0.004 [0.004, 0.004] | 0.660 [0.650, 0.666] | 0.508 [0.397, 0.543] |
| 0.1 | 0.177 [0.175, 0.179] | 0.004 [0.004, 0.005] | 0.664 [0.651, 0.673] | 0.594 [0.498, 0.611] |
| 0.2 | 0.167 [0.164, 0.170] | 0.005 [0.005, 0.005] | 0.672 [0.657, 0.682] | 0.679 [0.609, 0.703] |
| 0.3 | 0.154 [0.151, 0.156] | 0.005 [0.005, 0.006] | 0.687 [0.672, 0.696] | 0.765 [0.714, 0.780] |
| 0.4 | 0.138 [0.134, 0.140] | 0.006 [0.006, 0.007] | 0.705 [0.698, 0.715] | 0.838 [0.801, 0.859] |
| 0.5 | 0.121 [0.118, 0.123] | 0.008 [0.008, 0.008] | 0.727 [0.722, 0.740] | 0.902 [0.886, 0.910] |

![HHI](pilot_c_hhi.png)

![Cong. Capital Share](pilot_c_ccs_vs_alpha.png)

### Floor-hit clustering

Index of dispersion D of floor hits per market-period (Var/Mean over post-burn-in periods, median over markets) in the 27 traced runs, against the independent-firm binomial reference D = 1 − p; share of market-periods with ≥ 3 hits against its binomial reference; mean over reps.
Source: `pilot_c/burstiness_summary.csv` (`analysis/burstiness.py`)

| Family | α | D_median | D_binomial | Share ≥3 hits | Share ≥3 (binomial) | Max hits/period |
|--------|---|----------|------------|---------------|---------------------|-----------------|
| normal | 0.05 | 3.077 | 0.916 | 0.6185 | 0.7691 | 45.3 |
| normal | 0.1 | 2.983 | 0.920 | 0.5955 | 0.7419 | 45.3 |
| normal | 0.3 | 2.604 | 0.938 | 0.4798 | 0.5872 | 42.7 |
| laplace | 0.05 | 3.522 | 0.890 | 0.7350 | 0.8904 | 47.3 |
| laplace | 0.1 | 3.446 | 0.894 | 0.7186 | 0.8747 | 47.7 |
| laplace | 0.3 | 3.043 | 0.913 | 0.6381 | 0.7838 | 45.0 |
| t3 | 0.05 | 4.573 | 0.876 | 0.7560 | 0.9272 | 49.0 |
| t3 | 0.1 | 4.540 | 0.880 | 0.7430 | 0.9174 | 49.0 |
| t3 | 0.3 | 3.976 | 0.898 | 0.6855 | 0.8631 | 46.7 |

## Equal-split Check

Comparison of equal-split block to main normal block.
Source: `pilot_c/tidy.csv` column `K_post_burnin_median`, filtered by `block` and `sharing_rule`

### At α = 0.1

| Cost | K (proportional) [25,75] | K (equal) [25,75] |
|------|--------------------------|-------------------|
| exponential | 2.000 [2.000, 2.000] | 2.000 [2.000, 2.000] |
| linear | 3.000 [3.000, 3.000] | 2.000 [2.000, 2.000] |
| power_law | 3.000 [3.000, 3.000] | 2.000 [2.000, 2.000] |
| quadratic | 3.000 [3.000, 3.000] | 2.000 [2.000, 2.000] |

### Limited liability by α (C34)

Means over cost forms and replications. Capped members: members whose pooled factor hit the cap. Shortfall: capital charged to the other members of their unit (level units, market mean firm size = 1). Post-burn-in means per period; runs with capped members counted over t ≥ burn_in.
Source: `pilot_c/tidy.csv` columns `K_mean`, `cong_capital_share_median`, `acceptance_rate`, `pool_ruin_per_period`, `pool_shortfall_per_period`, block `equal-split`

| α | K (mean) | Pooled capital share | Acceptance rate | Capped members / period | Shortfall / period | Runs with capped members |
|---|----------|----------------------|-----------------|-------------------------|--------------------|--------------------------|
| 0.0 | n/a | 0.000 | 0.0000 | 0.00000 | 0.00000 | 0 of 20 |
| 0.01 | 2.112 | 0.082 | 0.1024 | 0.00000 | 0.00000 | 0 of 20 |
| 0.02 | 2.160 | 0.121 | 0.1481 | 0.00000 | 0.00000 | 0 of 20 |
| 0.05 | 2.233 | 0.155 | 0.1725 | 0.00003 | 0.00000 | 2 of 20 |
| 0.1 | 2.303 | 0.176 | 0.1896 | 0.00095 | 0.00003 | 19 of 20 |
| 0.2 | 2.366 | 0.192 | 0.1977 | 0.00623 | 0.00032 | 20 of 20 |
| 0.3 | 2.391 | 0.195 | 0.1915 | 0.01573 | 0.00103 | 20 of 20 |
| 0.4 | 2.416 | 0.192 | 0.1819 | 0.02793 | 0.00198 | 20 of 20 |
| 0.5 | 2.422 | 0.184 | 0.1709 | 0.04520 | 0.00378 | 20 of 20 |

## Cost Level

Φ(K) scaled at every K, laplace family: multipliers 0.5 and 2 for power_law and exponential from block `cost-level` (c0 scales the schedule), for linear and quadratic from block `cost-level-full` (c0, c1, c2); multiplier 1 from the main block.
Seeds are shared across the two multipliers within linear and within quadratic. Multiplier 1, and each multiplier of power_law and exponential, has its own seeds.
The linear and quadratic rows of block `cost-level` move only the intercept of Φ and are not used here.
Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `K_mean`, `cong_capital_share_median`, `hill_exponent_median`, medians across replications

### Mean K by multiplier

| Cost | α | ×0.5 | ×1 | ×2 | ×0.5 − ×1 | ×2 − ×1 |
|------|---|------|----|----|-----------|---------|
| linear | 0.1 | 3.16 | 2.92 | 2.71 | +0.24 | -0.21 |
| linear | 0.3 | 4.36 | 3.94 | 3.58 | +0.42 | -0.36 |
| linear | 0.5 | 6.72 | 5.84 | 5.09 | +0.88 | -0.75 |
| quadratic | 0.1 | 3.38 | 3.28 | 3.10 | +0.10 | -0.18 |
| quadratic | 0.3 | 4.53 | 4.41 | 4.09 | +0.11 | -0.32 |
| quadratic | 0.5 | 6.90 | 6.50 | 5.90 | +0.40 | -0.60 |
| exponential | 0.1 | 3.20 | 2.85 | 2.49 | +0.35 | -0.36 |
| exponential | 0.3 | 4.59 | 4.32 | 3.69 | +0.27 | -0.63 |
| exponential | 0.5 | 7.13 | 6.75 | 5.89 | +0.39 | -0.86 |
| power_law | 0.1 | 3.19 | 3.00 | 2.79 | +0.19 | -0.21 |
| power_law | 0.3 | 4.33 | 4.00 | 3.67 | +0.34 | -0.33 |
| power_law | 0.5 | 6.63 | 5.95 | 5.19 | +0.68 | -0.76 |

### By cost type, multiplier and α

| Cost | Multiplier | α | K median | K mean | CCS | Hill |
|------|------------|---|----------|--------|-----|------|
| exponential | 0.5 | 0 | nan | nan | 0.000 | 1.064 |
| exponential | 0.5 | 0.01 | 2.00 | 2.073 | 0.006 | 1.065 |
| exponential | 0.5 | 0.02 | 2.00 | 2.280 | 0.080 | 1.061 |
| exponential | 0.5 | 0.05 | 2.00 | 2.778 | 0.292 | 1.065 |
| exponential | 0.5 | 0.1 | 3.00 | 3.196 | 0.454 | 1.082 |
| exponential | 0.5 | 0.2 | 3.00 | 3.793 | 0.590 | 1.114 |
| exponential | 0.5 | 0.3 | 3.00 | 4.588 | 0.706 | 1.167 |
| exponential | 0.5 | 0.4 | 4.00 | 5.674 | 0.817 | 1.240 |
| exponential | 0.5 | 0.5 | 5.00 | 7.134 | 0.897 | 1.330 |
| exponential | 1 | 0 | nan | nan | 0.000 | 1.066 |
| exponential | 1 | 0.01 | 2.00 | 2.000 | 0.000 | 1.066 |
| exponential | 1 | 0.02 | 2.00 | 2.067 | 0.004 | 1.068 |
| exponential | 1 | 0.05 | 2.00 | 2.400 | 0.111 | 1.060 |
| exponential | 1 | 0.1 | 2.00 | 2.846 | 0.276 | 1.070 |
| exponential | 1 | 0.2 | 3.00 | 3.499 | 0.461 | 1.098 |
| exponential | 1 | 0.3 | 3.00 | 4.321 | 0.591 | 1.140 |
| exponential | 1 | 0.4 | 4.00 | 5.384 | 0.729 | 1.217 |
| exponential | 1 | 0.5 | 5.00 | 6.747 | 0.855 | 1.331 |
| exponential | 2 | 0 | nan | nan | 0.000 | 1.057 |
| exponential | 2 | 0.01 | nan | nan | 0.000 | 1.057 |
| exponential | 2 | 0.02 | 2.00 | 2.000 | 0.000 | 1.057 |
| exponential | 2 | 0.05 | 2.00 | 2.143 | 0.021 | 1.053 |
| exponential | 2 | 0.1 | 2.00 | 2.488 | 0.141 | 1.046 |
| exponential | 2 | 0.2 | 3.00 | 3.061 | 0.336 | 1.060 |
| exponential | 2 | 0.3 | 3.00 | 3.689 | 0.463 | 1.099 |
| exponential | 2 | 0.4 | 3.00 | 4.605 | 0.601 | 1.174 |
| exponential | 2 | 0.5 | 4.00 | 5.890 | 0.749 | 1.265 |
| linear | 0.5 | 0 | nan | nan | 0.000 | 1.071 |
| linear | 0.5 | 0.01 | 2.00 | 2.377 | 0.364 | 1.069 |
| linear | 0.5 | 0.02 | 2.00 | 2.587 | 0.477 | 1.069 |
| linear | 0.5 | 0.05 | 2.00 | 2.886 | 0.583 | 1.085 |
| linear | 0.5 | 0.1 | 3.00 | 3.159 | 0.654 | 1.094 |
| linear | 0.5 | 0.2 | 3.00 | 3.661 | 0.726 | 1.126 |
| linear | 0.5 | 0.3 | 3.00 | 4.357 | 0.791 | 1.175 |
| linear | 0.5 | 0.4 | 4.00 | 5.339 | 0.874 | 1.253 |
| linear | 0.5 | 0.5 | 5.00 | 6.721 | 0.919 | 1.316 |
| linear | 1 | 0 | nan | nan | 0.000 | 1.058 |
| linear | 1 | 0.01 | 2.00 | 2.184 | 0.224 | 1.057 |
| linear | 1 | 0.02 | 2.00 | 2.358 | 0.343 | 1.059 |
| linear | 1 | 0.05 | 2.00 | 2.651 | 0.488 | 1.061 |
| linear | 1 | 0.1 | 3.00 | 2.917 | 0.574 | 1.071 |
| linear | 1 | 0.2 | 3.00 | 3.390 | 0.670 | 1.095 |
| linear | 1 | 0.3 | 3.00 | 3.935 | 0.743 | 1.156 |
| linear | 1 | 0.4 | 4.00 | 4.674 | 0.824 | 1.203 |
| linear | 1 | 0.5 | 5.00 | 5.837 | 0.895 | 1.300 |
| linear | 2 | 0 | nan | nan | 0.000 | 1.071 |
| linear | 2 | 0.01 | 2.00 | 2.058 | 0.110 | 1.068 |
| linear | 2 | 0.02 | 2.00 | 2.194 | 0.240 | 1.068 |
| linear | 2 | 0.05 | 2.00 | 2.463 | 0.407 | 1.070 |
| linear | 2 | 0.1 | 2.00 | 2.708 | 0.509 | 1.086 |
| linear | 2 | 0.2 | 3.00 | 3.088 | 0.623 | 1.124 |
| linear | 2 | 0.3 | 3.00 | 3.580 | 0.701 | 1.159 |
| linear | 2 | 0.4 | 4.00 | 4.211 | 0.789 | 1.205 |
| linear | 2 | 0.5 | 4.00 | 5.090 | 0.873 | 1.311 |
| power_law | 0.5 | 0 | nan | nan | 0.000 | 1.056 |
| power_law | 0.5 | 0.01 | 2.00 | 2.433 | 0.435 | 1.061 |
| power_law | 0.5 | 0.02 | 2.00 | 2.646 | 0.523 | 1.058 |
| power_law | 0.5 | 0.05 | 2.00 | 2.917 | 0.611 | 1.072 |
| power_law | 0.5 | 0.1 | 3.00 | 3.190 | 0.666 | 1.089 |
| power_law | 0.5 | 0.2 | 3.00 | 3.692 | 0.744 | 1.112 |
| power_law | 0.5 | 0.3 | 3.00 | 4.335 | 0.809 | 1.172 |
| power_law | 0.5 | 0.4 | 4.00 | 5.361 | 0.878 | 1.230 |
| power_law | 0.5 | 0.5 | 5.00 | 6.630 | 0.919 | 1.326 |
| power_law | 1 | 0 | nan | nan | 0.000 | 1.064 |
| power_law | 1 | 0.01 | 2.00 | 2.287 | 0.365 | 1.067 |
| power_law | 1 | 0.02 | 2.00 | 2.469 | 0.459 | 1.069 |
| power_law | 1 | 0.05 | 2.00 | 2.747 | 0.563 | 1.075 |
| power_law | 1 | 0.1 | 3.00 | 3.005 | 0.634 | 1.098 |
| power_law | 1 | 0.2 | 3.00 | 3.458 | 0.703 | 1.113 |
| power_law | 1 | 0.3 | 3.00 | 3.999 | 0.782 | 1.165 |
| power_law | 1 | 0.4 | 4.00 | 4.798 | 0.857 | 1.220 |
| power_law | 1 | 0.5 | 5.00 | 5.948 | 0.906 | 1.329 |
| power_law | 2 | 0 | nan | nan | 0.000 | 1.056 |
| power_law | 2 | 0.01 | 2.00 | 2.118 | 0.217 | 1.060 |
| power_law | 2 | 0.02 | 2.00 | 2.261 | 0.342 | 1.060 |
| power_law | 2 | 0.05 | 2.00 | 2.547 | 0.470 | 1.061 |
| power_law | 2 | 0.1 | 2.00 | 2.795 | 0.563 | 1.079 |
| power_law | 2 | 0.2 | 3.00 | 3.191 | 0.658 | 1.106 |
| power_law | 2 | 0.3 | 3.00 | 3.667 | 0.741 | 1.171 |
| power_law | 2 | 0.4 | 4.00 | 4.290 | 0.816 | 1.211 |
| power_law | 2 | 0.5 | 4.00 | 5.190 | 0.884 | 1.295 |
| quadratic | 0.5 | 0 | nan | nan | 0.000 | 1.060 |
| quadratic | 0.5 | 0.01 | 2.00 | 2.733 | 0.413 | 1.059 |
| quadratic | 0.5 | 0.02 | 2.00 | 2.924 | 0.511 | 1.058 |
| quadratic | 0.5 | 0.05 | 3.00 | 3.156 | 0.595 | 1.067 |
| quadratic | 0.5 | 0.1 | 3.00 | 3.382 | 0.664 | 1.081 |
| quadratic | 0.5 | 0.2 | 3.00 | 3.849 | 0.742 | 1.130 |
| quadratic | 0.5 | 0.3 | 3.00 | 4.525 | 0.819 | 1.181 |
| quadratic | 0.5 | 0.4 | 4.00 | 5.541 | 0.881 | 1.225 |
| quadratic | 0.5 | 0.5 | 5.00 | 6.900 | 0.920 | 1.315 |
| quadratic | 1 | 0 | nan | nan | 0.000 | 1.056 |
| quadratic | 1 | 0.01 | 2.00 | 2.489 | 0.240 | 1.054 |
| quadratic | 1 | 0.02 | 2.00 | 2.755 | 0.395 | 1.054 |
| quadratic | 1 | 0.05 | 3.00 | 3.036 | 0.531 | 1.062 |
| quadratic | 1 | 0.1 | 3.00 | 3.281 | 0.606 | 1.085 |
| quadratic | 1 | 0.2 | 3.00 | 3.768 | 0.697 | 1.117 |
| quadratic | 1 | 0.3 | 4.00 | 4.412 | 0.779 | 1.174 |
| quadratic | 1 | 0.4 | 4.00 | 5.313 | 0.861 | 1.226 |
| quadratic | 1 | 0.5 | 5.00 | 6.498 | 0.912 | 1.313 |
| quadratic | 2 | 0 | nan | nan | 0.000 | 1.060 |
| quadratic | 2 | 0.01 | 2.00 | 2.275 | 0.118 | 1.062 |
| quadratic | 2 | 0.02 | 2.00 | 2.531 | 0.276 | 1.061 |
| quadratic | 2 | 0.05 | 2.00 | 2.841 | 0.438 | 1.067 |
| quadratic | 2 | 0.1 | 3.00 | 3.099 | 0.554 | 1.072 |
| quadratic | 2 | 0.2 | 3.00 | 3.529 | 0.657 | 1.114 |
| quadratic | 2 | 0.3 | 3.00 | 4.089 | 0.743 | 1.170 |
| quadratic | 2 | 0.4 | 4.00 | 4.883 | 0.827 | 1.226 |
| quadratic | 2 | 0.5 | 5.00 | 5.902 | 0.900 | 1.319 |

## Lookback Sensitivity

Lookback block (l = 20, 50, 100, 200, 500) by family and α; reference row: main laplace/power_law cell (l = 50).
Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `K_mean`, `cong_capital_share_median`, `floor_exits_per_period`, `hill_exponent_median`, filtered by `block` and `lookback`
Figure: `diagnostics/pilot_c_lookback_k.png`

### At α = 0.1

| Family | Lookback | K [25,75] | K_mean | CCS | Floor exits/period | Hill [25,75] |
|--------|----------|-----------|--------|-----|--------------------|--------------|
| laplace (main ref) | 50 | 3.000 [3.000, 3.000] | — | — | — | 1.098 [1.089, 1.100] |
| laplace | 20 | 2.000 [2.000, 2.000] | 2.660 | 0.428 | 44.42 | 1.075 [1.074, 1.083] |
| laplace | 50 | 3.000 [3.000, 3.000] | 3.005 | 0.636 | 45.92 | 1.085 [1.082, 1.088] |
| laplace | 100 | 3.000 [3.000, 3.000] | 3.194 | 0.756 | 47.50 | 1.098 [1.093, 1.098] |
| laplace | 200 | 3.000 [3.000, 3.000] | 3.453 | 0.825 | 44.46 | 1.086 [1.086, 1.093] |
| laplace | 500 | 4.000 [3.000, 4.000] | 3.921 | 0.881 | 49.81 | 1.087 [1.071, 1.098] |
| t3 | 20 | 2.000 [2.000, 2.000] | 2.562 | 0.430 | 54.26 | 1.009 [0.997, 1.010] |
| t3 | 50 | 2.000 [2.000, 2.000] | 2.873 | 0.631 | 50.74 | 1.021 [1.015, 1.026] |
| t3 | 100 | 3.000 [3.000, 3.000] | 3.085 | 0.741 | 50.53 | 1.033 [1.024, 1.047] |
| t3 | 200 | 3.000 [3.000, 3.000] | 3.268 | 0.814 | 55.45 | 1.020 [1.015, 1.024] |
| t3 | 500 | 3.000 [3.000, 3.000] | 3.748 | 0.852 | 57.88 | 1.023 [1.008, 1.038] |

### At α = 0.3

| Family | Lookback | K [25,75] | K_mean | CCS | Floor exits/period | Hill [25,75] |
|--------|----------|-----------|--------|-----|--------------------|--------------|
| laplace (main ref) | 50 | 3.000 [3.000, 3.000] | — | — | — | 1.165 [1.154, 1.174] |
| laplace | 20 | 3.000 [3.000, 3.000] | 3.286 | 0.600 | 36.77 | 1.119 [1.118, 1.124] |
| laplace | 50 | 3.000 [3.000, 3.000] | 4.038 | 0.785 | 36.00 | 1.171 [1.155, 1.177] |
| laplace | 100 | 4.000 [4.000, 4.000] | 4.419 | 0.863 | 36.94 | 1.175 [1.162, 1.179] |
| laplace | 200 | 4.000 [4.000, 4.000] | 4.821 | 0.909 | 34.47 | 1.198 [1.197, 1.203] |
| laplace | 500 | 5.000 [5.000, 5.000] | 5.361 | 0.917 | 38.38 | 1.184 [1.176, 1.186] |
| t3 | 20 | 2.000 [2.000, 2.000] | 3.005 | 0.589 | 46.86 | 1.041 [1.039, 1.053] |
| t3 | 50 | 3.000 [3.000, 3.000] | 3.682 | 0.775 | 41.98 | 1.093 [1.093, 1.100] |
| t3 | 100 | 3.000 [3.000, 3.000] | 4.067 | 0.856 | 41.09 | 1.094 [1.086, 1.104] |
| t3 | 200 | 4.000 [4.000, 4.000] | 4.282 | 0.889 | 45.66 | 1.087 [1.081, 1.090] |
| t3 | 500 | 4.000 [4.000, 4.000] | 4.920 | 0.893 | 47.04 | 1.107 [1.101, 1.108] |

## Search intensity

Search block (merge_thresh = 0.2, 1.0) and the main power_law cell (merge_thresh = 0.05), by family × α × merge_thresh; medians over reps.
Source: `pilot_c/tidy.csv`, blocks `search` and `main` (power_law). Figure: `diagnostics/pilot_c_search.png` (K* dashed).

| Family | α | merge_thresh | K_mean | K_post_burnin_median | CCS | Mergers/period | Floor exits/period | Hill |
|--------|---|--------------|--------|----------------------|-----|----------------|--------------------|------|
| normal | 0.05 | 0.05 | 2.705 | 2.0 | 0.473 | 39.54 | 36.03 | 1.081 |
| normal | 0.05 | 0.2 | 3.245 | 3.0 | 0.687 | 74.13 | 70.20 | 1.090 |
| normal | 0.05 | 1 | 3.696 | 4.0 | 0.807 | 130.49 | 126.42 | 1.089 |
| normal | 0.1 | 0.05 | 3.040 | 3.0 | 0.561 | 37.70 | 34.76 | 1.087 |
| normal | 0.1 | 0.2 | 3.843 | 4.0 | 0.745 | 68.90 | 66.10 | 1.100 |
| normal | 0.1 | 1 | 4.511 | 4.0 | 0.856 | 121.39 | 118.76 | 1.111 |
| normal | 0.3 | 0.05 | 4.288 | 4.0 | 0.733 | 27.32 | 26.03 | 1.169 |
| normal | 0.3 | 0.2 | 5.942 | 6.0 | 0.872 | 45.37 | 44.59 | 1.202 |
| normal | 0.3 | 1 | 7.152 | 7.0 | 0.925 | 83.27 | 82.69 | 1.182 |
| normal | 0.5 | 0.05 | 6.337 | 6.0 | 0.912 | 16.29 | 16.06 | 1.316 |
| normal | 0.5 | 0.2 | 9.010 | 9.0 | 0.966 | 25.04 | 24.99 | 1.331 |
| normal | 0.5 | 1 | 10.451 | 11.0 | 0.984 | 46.56 | 46.53 | 1.322 |
| laplace | 0.05 | 0.05 | 2.747 | 2.0 | 0.563 | 50.20 | 48.15 | 1.075 |
| laplace | 0.05 | 0.2 | 3.341 | 3.0 | 0.725 | 95.17 | 92.75 | 1.062 |
| laplace | 0.05 | 1 | 3.906 | 4.0 | 0.853 | 176.96 | 174.77 | 1.074 |
| laplace | 0.1 | 0.05 | 3.005 | 3.0 | 0.634 | 48.57 | 46.77 | 1.098 |
| laplace | 0.1 | 0.2 | 3.863 | 3.0 | 0.791 | 89.70 | 87.88 | 1.083 |
| laplace | 0.1 | 1 | 4.634 | 4.0 | 0.885 | 166.25 | 164.80 | 1.088 |
| laplace | 0.3 | 0.05 | 3.999 | 3.0 | 0.782 | 37.76 | 36.88 | 1.165 |
| laplace | 0.3 | 0.2 | 5.765 | 5.0 | 0.888 | 63.08 | 62.51 | 1.195 |
| laplace | 0.3 | 1 | 7.144 | 7.0 | 0.944 | 119.03 | 118.66 | 1.169 |
| laplace | 0.5 | 0.05 | 5.948 | 5.0 | 0.906 | 25.24 | 25.05 | 1.329 |
| laplace | 0.5 | 0.2 | 8.839 | 9.0 | 0.970 | 38.08 | 38.04 | 1.322 |
| laplace | 0.5 | 1 | 10.656 | 11.0 | 0.987 | 73.22 | 73.19 | 1.307 |
| t3 | 0.05 | 0.05 | 2.665 | 2.0 | 0.563 | 55.05 | 53.33 | 1.005 |
| t3 | 0.05 | 0.2 | 3.251 | 3.0 | 0.745 | 111.81 | 109.87 | 1.012 |
| t3 | 0.05 | 1 | 3.884 | 4.0 | 0.857 | 206.59 | 204.92 | 1.002 |
| t3 | 0.1 | 0.05 | 2.866 | 2.0 | 0.627 | 53.69 | 52.13 | 1.018 |
| t3 | 0.1 | 0.2 | 3.682 | 3.0 | 0.796 | 106.48 | 104.98 | 1.031 |
| t3 | 0.1 | 1 | 4.500 | 4.0 | 0.886 | 194.35 | 193.24 | 1.023 |
| t3 | 0.3 | 0.05 | 3.667 | 3.0 | 0.773 | 44.12 | 43.18 | 1.096 |
| t3 | 0.3 | 0.2 | 5.213 | 5.0 | 0.903 | 80.49 | 79.99 | 1.101 |
| t3 | 0.3 | 1 | 6.661 | 6.0 | 0.954 | 144.23 | 143.91 | 1.104 |
| t3 | 0.5 | 0.05 | 5.063 | 4.0 | 0.884 | 32.60 | 32.28 | 1.182 |
| t3 | 0.5 | 0.2 | 7.755 | 8.0 | 0.962 | 53.38 | 53.28 | 1.221 |
| t3 | 0.5 | 1 | 9.775 | 10.0 | 0.984 | 96.94 | 96.86 | 1.229 |

## Economy size

Size block (laplace, power_law, merge_thresh 0.05, floor_c = c*(N)) by (M, N) × α, and the main laplace/power_law cell (M = N = 50, floor_c = 0.12717) at the same α; medians over reps. K* from `analytics/benchmarks.csv` at the row's M.
Source: `pilot_c/tidy.csv`, blocks `size` and `main`. Figure: `diagnostics/pilot_c_size.png`.

| M | N | α | K_mean | K* | K_mean/K* | CCS | Hill | Floor-hit (standalone) | Floor exits/period | Mergers/period |
|---|---|---|--------|----|-----------|-----|------|------------------------|--------------------|----------------|
| 25 | 25 | 0 | — | — | — | 0.000 | 1.103 | 0.0891 | 0.00 | 0.00 |
| 25 | 25 | 0.1 | 3.039 | 10 | 0.304 | 0.623 | 1.119 | 0.0663 | 8.86 | 9.56 |
| 25 | 25 | 0.3 | 4.064 | 16 | 0.254 | 0.785 | 1.196 | 0.0512 | 6.55 | 6.91 |
| 25 | 25 | 0.5 | 6.205 | 20 | 0.310 | 0.940 | 1.403 | 0.0357 | 3.80 | 3.86 |
| 25 | 100 | 0 | — | — | — | 0.000 | 1.059 | 0.1316 | 0.00 | 0.00 |
| 25 | 100 | 0.1 | 2.869 | 10 | 0.287 | 0.608 | 1.083 | 0.1019 | 52.74 | 54.27 |
| 25 | 100 | 0.3 | 3.624 | 16 | 0.227 | 0.744 | 1.154 | 0.0869 | 43.81 | 44.66 |
| 25 | 100 | 0.5 | 4.830 | 20 | 0.242 | 0.871 | 1.246 | 0.0707 | 32.69 | 32.91 |
| 100 | 25 | 0 | — | — | — | 0.000 | 1.071 | 0.0850 | 0.00 | 0.00 |
| 100 | 25 | 0.1 | 3.131 | 10 | 0.313 | 0.611 | 1.102 | 0.0634 | 33.22 | 36.01 |
| 100 | 25 | 0.3 | 4.530 | 16 | 0.283 | 0.796 | 1.221 | 0.0485 | 23.62 | 24.95 |
| 100 | 25 | 0.5 | 7.502 | 20 | 0.375 | 0.949 | 1.456 | 0.0338 | 13.24 | 13.42 |
| 100 | 100 | 0 | — | — | — | 0.000 | 1.061 | 0.1341 | 0.00 | 0.00 |
| 100 | 100 | 0.1 | 2.971 | 10 | 0.297 | 0.613 | 1.079 | 0.1037 | 214.28 | 219.61 |
| 100 | 100 | 0.3 | 3.920 | 16 | 0.245 | 0.762 | 1.144 | 0.0881 | 174.19 | 176.84 |
| 100 | 100 | 0.5 | 5.749 | 20 | 0.287 | 0.885 | 1.266 | 0.0704 | 123.69 | 124.20 |
| 50 | 50 | 0 | — | — | — | 0.000 | 1.064 | 0.1160 | 0.00 | 0.00 |
| 50 | 50 | 0.1 | 3.005 | 10 | 0.300 | 0.634 | 1.098 | 0.0885 | 46.77 | 48.57 |
| 50 | 50 | 0.3 | 3.999 | 16 | 0.250 | 0.782 | 1.165 | 0.0732 | 36.88 | 37.76 |
| 50 | 50 | 0.5 | 5.948 | 20 | 0.297 | 0.906 | 1.329 | 0.0564 | 25.05 | 25.24 |

## Cost level high

Cost-level-high block (power law Φ(K) = c0·K^1.2 with c0 × 5, 10, 20, a level shift of Φ at every K; merge_thresh 0.05) and the main power_law cell (× 1), laplace and t3, by family × multiplier × α; medians over 5 reps. K* from `analytics/benchmarks.csv` (M = 50, median market IQR 0.2) at the row's multiplier; acceptance = mergers / proposals after burn-in.
Source: `pilot_c/tidy.csv`, blocks `cost-level-high` and `main`.

| Family | × c0 | α | K_mean | K* | K_mean/K* | CCS | Acceptance | Mergers/period | Floor exits/period |
|--------|------|---|--------|----|-----------|-----|------------|----------------|--------------------|
| laplace | 1 | 0.1 | 3.005 | 10 | 0.300 | 0.634 | 0.4131 | 48.57 | 46.77 |
| laplace | 1 | 0.3 | 3.999 | 16 | 0.250 | 0.782 | 0.3607 | 37.76 | 36.88 |
| laplace | 1 | 0.5 | 5.948 | 20 | 0.297 | 0.906 | 0.3206 | 25.24 | 25.05 |
| laplace | 5 | 0.1 | 2.480 | 5 | 0.496 | 0.438 | 0.3628 | 44.07 | 41.09 |
| laplace | 5 | 0.3 | 3.138 | 8 | 0.392 | 0.636 | 0.3139 | 36.09 | 34.18 |
| laplace | 5 | 0.5 | 4.129 | 9 | 0.459 | 0.815 | 0.2439 | 24.55 | 23.97 |
| laplace | 10 | 0.1 | 2.276 | 4 | 0.569 | 0.337 | 0.3343 | 41.06 | 37.99 |
| laplace | 10 | 0.3 | 2.760 | 6 | 0.460 | 0.536 | 0.3112 | 36.96 | 34.22 |
| laplace | 10 | 0.5 | 3.441 | 7 | 0.492 | 0.699 | 0.2409 | 26.75 | 25.46 |
| laplace | 20 | 0.1 | 2.120 | 3 | 0.707 | 0.208 | 0.2566 | 31.87 | 29.64 |
| laplace | 20 | 0.3 | 2.452 | 4 | 0.613 | 0.410 | 0.3063 | 37.31 | 33.80 |
| laplace | 20 | 0.5 | 2.847 | 5 | 0.569 | 0.566 | 0.2508 | 29.64 | 27.22 |
| t3 | 1 | 0.1 | 2.866 | 20 | 0.143 | 0.627 | 0.4520 | 53.69 | 52.13 |
| t3 | 1 | 0.3 | 3.667 | 47 | 0.078 | 0.773 | 0.4026 | 44.12 | 43.18 |
| t3 | 1 | 0.5 | 5.063 | 50 | 0.101 | 0.884 | 0.3594 | 32.60 | 32.28 |
| t3 | 5 | 0.1 | 2.491 | 7 | 0.356 | 0.496 | 0.4254 | 51.84 | 49.71 |
| t3 | 5 | 0.3 | 3.042 | 10 | 0.304 | 0.662 | 0.3786 | 44.13 | 42.69 |
| t3 | 5 | 0.5 | 3.881 | 13 | 0.299 | 0.814 | 0.3185 | 33.80 | 33.25 |
| t3 | 10 | 0.1 | 2.318 | 4 | 0.579 | 0.408 | 0.4001 | 49.12 | 46.72 |
| t3 | 10 | 0.3 | 2.766 | 7 | 0.395 | 0.585 | 0.3731 | 44.48 | 42.57 |
| t3 | 10 | 0.5 | 3.343 | 9 | 0.371 | 0.742 | 0.3102 | 34.98 | 34.02 |
| t3 | 20 | 0.1 | 2.161 | 3 | 0.720 | 0.294 | 0.3408 | 42.12 | 40.00 |
| t3 | 20 | 0.3 | 2.496 | 5 | 0.499 | 0.477 | 0.3679 | 44.61 | 42.05 |
| t3 | 20 | 0.5 | 2.884 | 6 | 0.481 | 0.631 | 0.3125 | 36.81 | 35.16 |

**Where scope starts to fall, and whether cost binds at K\* (C35).** There is no threshold: scope falls at every
cost level tried, by about 0.35 in mean K per doubling of c0.

- For laplace at α = 0.3, mean K (medians over reps) is 4.33, 4.00 and 3.67 at ×0.5, ×1 and ×2 (C33 `cost-level`
  block and the main cell, which use different seeds), then 3.14, 2.76 and 2.45 at ×5, ×10 and ×20.
- For t3 at α = 0.3 it is 3.67, 3.04, 2.77 and 2.50 at ×1, 5, 10 and 20.

The fall comes through acceptance at every K ≥ 3. For laplace at α = 0.3 and K = 4, acceptance is 0.41, 0.28,
0.20 and 0.11 at ×1, 5, 10 and 20 (`pilot_c/acceptance_by_K_pooled.csv`). Mergers per period barely move, because
more standalones make more proposals.

K\* falls faster than K: K/K\* rises from 0.25 (laplace) and 0.08 (t3) at ×1 to 0.61 and 0.50 at ×20. At ×20, K\*
(4 for laplace, 5 for t3) has come down to the scope that forms at ×1, and the analytical marginal cost of the
K-th member is close to its marginal gain at K ≈ 4–5. Yet K is 2.45–2.50. This is still half to two thirds of
K\*, against 0.7 at α = 0.1, where K\* = 3.

Cost therefore binds at the margin from the lowest level on, but even at ×20 it does not take over: K does not
settle at K\*, and attrition and consent keep it below. (K_mean counts units only, so it cannot fall below 2.)

## Owner-return growth by status

Next-period owner return (log share change net of floor recapitalisation) of members minus standalones within (market, size decile) cells, weighted by member firm-periods; all firm-periods after burn-in of the 27 traced main-block power_law scenarios; mean over reps 0-2. Status is taken at the start of the period. Primary result; the event study below is secondary.
Source: `pilot_c/status_panel_summary.csv`. Figure: `diagnostics/pilot_c_status_panel.png`

| Family | α | diff_fe_mean | diff_fe_median | diff_K2_mean | diff_K3_4_mean | diff_K5p_mean | floor_rate_member | floor_rate_standalone | inc_member_mean | inc_standalone_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | 0.00054 | -0.00003 | 0.00043 | 0.00060 | 0.00070 | 0.02029 | 0.15227 | -0.01046 | -0.01090 |
| normal | 0.1 | 0.00131 | 0.00045 | 0.00089 | 0.00136 | 0.00176 | 0.01805 | 0.16238 | -0.00953 | -0.01085 |
| normal | 0.3 | 0.00427 | 0.00242 | 0.00267 | 0.00406 | 0.00474 | 0.01159 | 0.17325 | -0.00629 | -0.01053 |
| laplace | 0.05 | 0.00103 | 0.00015 | 0.00087 | 0.00109 | 0.00126 | 0.02745 | 0.18919 | -0.01835 | -0.01962 |
| laplace | 0.1 | 0.00272 | 0.00084 | 0.00204 | 0.00288 | 0.00335 | 0.02499 | 0.19614 | -0.01671 | -0.01963 |
| laplace | 0.3 | 0.00815 | 0.00378 | 0.00560 | 0.00769 | 0.00920 | 0.01704 | 0.20643 | -0.01131 | -0.01935 |
| t3 | 0.05 | 0.00266 | 0.00028 | 0.00233 | 0.00278 | 0.00342 | 0.03206 | 0.20114 | -0.02300 | -0.02590 |
| t3 | 0.1 | 0.00452 | 0.00067 | 0.00379 | 0.00471 | 0.00544 | 0.02955 | 0.20713 | -0.02096 | -0.02582 |
| t3 | 0.3 | 0.01136 | 0.00358 | 0.00837 | 0.01106 | 0.01296 | 0.02121 | 0.21750 | -0.01445 | -0.02563 |

| Family | α | diff_dec1_mean | diff_dec2_mean | diff_dec3_mean | diff_dec4_mean | diff_dec5_mean | diff_dec6_mean | diff_dec7_mean | diff_dec8_mean | diff_dec9_mean | diff_dec10_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | 0.00080 | 0.00032 | 0.00080 | 0.00061 | 0.00059 | 0.00043 | 0.00055 | 0.00027 | 0.00092 | 0.00036 |
| normal | 0.1 | 0.00120 | 0.00168 | 0.00138 | 0.00134 | 0.00138 | 0.00149 | 0.00127 | 0.00159 | 0.00128 | 0.00054 |
| normal | 0.3 | 0.00405 | 0.00456 | 0.00484 | 0.00470 | 0.00483 | 0.00455 | 0.00482 | 0.00503 | 0.00300 | 0.00255 |
| laplace | 0.05 | 0.00068 | 0.00161 | 0.00172 | 0.00116 | 0.00090 | 0.00109 | 0.00109 | 0.00113 | 0.00079 | 0.00057 |
| laplace | 0.1 | 0.00182 | 0.00272 | 0.00328 | 0.00288 | 0.00280 | 0.00313 | 0.00308 | 0.00209 | 0.00327 | 0.00164 |
| laplace | 0.3 | 0.00501 | 0.00890 | 0.00866 | 0.00863 | 0.00850 | 0.00912 | 0.00853 | 0.00790 | 0.00872 | 0.00555 |
| t3 | 0.05 | 0.00211 | 0.00265 | 0.00270 | 0.00332 | 0.00259 | 0.00311 | 0.00283 | 0.00327 | 0.00230 | 0.00145 |
| t3 | 0.1 | 0.00037 | 0.00373 | 0.00581 | 0.00464 | 0.00496 | 0.00553 | 0.00523 | 0.00443 | 0.00442 | 0.00247 |
| t3 | 0.3 | 0.00749 | 0.01106 | 0.01184 | 0.01197 | 0.01251 | 0.01196 | 0.01215 | 0.01239 | 0.01086 | 0.00840 |

## Event study (ITT and owner path)

Conglomerate entry, main block power_law, 27 traced scenarios; mean over reps 0-2 of per-scenario medians. ITT = all matched entries on raw log share; owner = the same DiD on log share net of cumulative floor jumps (what owners earned, excluding recapitalisation).
Source: `pilot_c/event_study_summary.csv`

| Family | α | n_events | n_matched | share_floor_before | did_itt_median | did_owner_median | did_nofloor_median | did_owner_nofloor_median | did_small | did_owner_small | did_mid | did_owner_mid | did_large | did_owner_large |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | 470901 | 243728 | 0.75352 | 0.00295 | 0.00085 | -0.00198 | -0.00235 | 0.00434 | 0.00061 | 0.00288 | 0.00123 | 0.00120 | 0.00071 |
| normal | 0.1 | 429736 | 208214 | 0.78225 | 0.00312 | 0.00151 | 0.00114 | 0.00103 | 0.00441 | 0.00095 | 0.00287 | 0.00140 | 0.00181 | 0.00211 |
| normal | 0.3 | 289978 | 138034 | 0.84619 | 0.00240 | 0.00195 | 0.00294 | 0.00331 | 0.00388 | 0.00063 | 0.00203 | 0.00170 | 0.00110 | 0.00354 |
| laplace | 0.05 | 563595 | 270911 | 0.83302 | 0.00384 | -0.00011 | -0.00185 | -0.00221 | 0.00495 | -0.00137 | 0.00376 | 0.00007 | 0.00261 | 0.00097 |
| laplace | 0.1 | 528041 | 244922 | 0.85079 | 0.00390 | 0.00065 | 0.00195 | 0.00174 | 0.00505 | -0.00080 | 0.00362 | 0.00021 | 0.00284 | 0.00250 |
| laplace | 0.3 | 390683 | 177493 | 0.89460 | 0.00341 | 0.00127 | 0.00495 | 0.00579 | 0.00472 | -0.00084 | 0.00297 | 0.00069 | 0.00242 | 0.00387 |
| t3 | 0.05 | 616212 | 306120 | 0.85482 | 0.00357 | -0.00075 | -0.00212 | -0.00289 | 0.00463 | -0.00184 | 0.00332 | -0.00080 | 0.00257 | 0.00042 |
| t3 | 0.1 | 585135 | 280712 | 0.86717 | 0.00368 | -0.00055 | 0.00160 | 0.00149 | 0.00474 | -0.00172 | 0.00337 | -0.00106 | 0.00283 | 0.00109 |
| t3 | 0.3 | 458074 | 216788 | 0.90298 | 0.00344 | -0.00037 | 0.00453 | 0.00571 | 0.00465 | -0.00212 | 0.00302 | -0.00126 | 0.00253 | 0.00211 |

**Means (heavy tails: the median of a 50-period window can hide rare large losses) and the joiner's owner path**

| Family | α | did_itt_mean | did_owner_mean | did_owner_trim_mean | joiner_before_owner | joiner_after_owner | joiner_before_owner_mean | joiner_after_owner_mean |
|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | 0.00298 | 0.00074 | 0.00077 | -0.01709 | -0.00960 | -0.01824 | -0.01104 |
| normal | 0.1 | 0.00338 | 0.00134 | 0.00137 | -0.01778 | -0.00901 | -0.01892 | -0.01052 |
| normal | 0.3 | 0.00249 | 0.00180 | 0.00182 | -0.01836 | -0.00655 | -0.01946 | -0.00840 |
| laplace | 0.05 | 0.00420 | -0.00039 | -0.00036 | -0.02684 | -0.01766 | -0.02900 | -0.02012 |
| laplace | 0.1 | 0.00436 | 0.00028 | 0.00032 | -0.02739 | -0.01673 | -0.02964 | -0.01926 |
| laplace | 0.3 | 0.00365 | 0.00093 | 0.00098 | -0.02771 | -0.01287 | -0.02995 | -0.01579 |
| t3 | 0.05 | 0.00397 | -0.00144 | -0.00123 | -0.03079 | -0.02152 | -0.03545 | -0.02558 |
| t3 | 0.1 | 0.00425 | -0.00131 | -0.00109 | -0.03123 | -0.02047 | -0.03575 | -0.02444 |
| t3 | 0.3 | 0.00379 | -0.00149 | -0.00109 | -0.03105 | -0.01642 | -0.03552 | -0.02056 |

**Owner DiD by conglomerate size at entry (K = 2, 3–4, ≥ 5)**

| Family | α | n_K2 | n_K3_4 | n_K5p | did_owner_K2_median | did_owner_K2_mean | did_owner_K3_4_median | did_owner_K3_4_mean | did_owner_K5p_median | did_owner_K5p_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | 175155 | 60361 | 8212 | 0.00025 | 0.00004 | 0.00219 | 0.00211 | 0.00577 | 0.00548 |
| normal | 0.1 | 138638 | 52748 | 16829 | 0.00063 | 0.00043 | 0.00276 | 0.00239 | 0.00600 | 0.00550 |
| normal | 0.3 | 79139 | 28185 | 30710 | 0.00054 | 0.00028 | 0.00241 | 0.00207 | 0.00581 | 0.00545 |
| laplace | 0.05 | 193741 | 65367 | 11803 | -0.00087 | -0.00127 | 0.00146 | 0.00119 | 0.00549 | 0.00536 |
| laplace | 0.1 | 165451 | 59321 | 20150 | -0.00044 | -0.00089 | 0.00216 | 0.00172 | 0.00633 | 0.00560 |
| laplace | 0.3 | 105030 | 36301 | 36162 | -0.00051 | -0.00099 | 0.00202 | 0.00156 | 0.00671 | 0.00591 |
| t3 | 0.05 | 221734 | 71755 | 12631 | -0.00168 | -0.00234 | 0.00154 | 0.00054 | 0.00483 | 0.00320 |
| t3 | 0.1 | 193514 | 66422 | 20776 | -0.00165 | -0.00243 | 0.00141 | 0.00054 | 0.00496 | 0.00312 |
| t3 | 0.3 | 132850 | 46003 | 37935 | -0.00185 | -0.00281 | 0.00077 | -0.00056 | 0.00442 | 0.00205 |

## Correlation Sensitivity

Comparison of correlation block (cross_corr=0.3) to main laplace/power_law cell (cross_corr=0).
Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `hill_exponent_median`, filtered by `block` and `cross_corr`

### At α = 0.1

| Cross-corr | K [25,75] | Hill [25,75] |
|------------|-----------|--------------|
| 0.0 (ref) | 3.000 [3.000, 3.000] | 1.098 [1.089, 1.100] |
| 0.3 | 3.000 [3.000, 3.000] | 1.091 [1.080, 1.097] |

## Endogenous α

Analysis of endogenous-alpha block.
Source: `pilot_c/tidy.csv` columns `alpha_adopted_median`, `alpha_adopted_mean`, `alpha_adopted_std`

### Adopted α by family and cost

| Family | Cost | α adopted (median) | α adopted (mean) | K |
|--------|------|--------------------|------------------|---|
| laplace | exponential | 0.850 | 0.813 | 13.0 |
| laplace | linear | 0.850 | 0.803 | 8.0 |
| laplace | power_law | 0.800 | 0.803 | 9.0 |
| laplace | quadratic | 0.800 | 0.802 | 10.0 |
| normal | exponential | 0.800 | 0.816 | 22.0 |
| normal | linear | 0.850 | 0.821 | 11.0 |
| normal | power_law | 0.850 | 0.818 | 11.0 |
| normal | quadratic | 0.850 | 0.818 | 13.0 |
| t3 | exponential | 0.800 | 0.786 | 5.0 |
| t3 | linear | 0.800 | 0.786 | 5.0 |
| t3 | power_law | 0.800 | 0.782 | 5.0 |
| t3 | quadratic | 0.800 | 0.790 | 5.0 |

![Endogenous α histogram](pilot_c_endo_alpha_hist.png)

### α and member volatility (one row per active unit at T)

Spearman rank correlation of the unit's time-averaged post-burn-in α (`alpha_tavg`) with the sd and the mean of its member markets' IQR.
Source: `pilot_c/alpha_scatter.csv`. Figure: `diagnostics/pilot_c_alpha_scatter.png`

| Family | n | ρ(α, sd_iqr) | p | ρ(α, mean_iqr) | p |
|--------|---|--------------|---|----------------|---|
| normal | 3210 | 0.040 | 0.0229 | 0.260 | 6.39e-51 |
| laplace | 3353 | 0.053 | 0.00214 | 0.194 | 8.97e-30 |
| t3 | 4166 | 0.069 | 9.3e-06 | 0.139 | 2.51e-19 |

![α vs sd of member IQR](pilot_c_alpha_scatter.png)

## Assortativity

SD of member IQR / SD of random K-subset. Ratio < 1 indicates positive assortment
(conglomerate members have more similar growth volatilities than random groups).

Source: `pilot_c/tidy.csv` column `assort_iqr`

### At α = 0.1 (power_law cost)

| Family | Assort. ratio [25,75] |
|--------|----------------------|
| laplace | 0.542 [0.534, 0.546] |
| normal | 0.541 [0.536, 0.551] |
| t3 | 0.562 [0.556, 0.568] |

![Assortativity](pilot_c_assortativity.png)

## Runtime Statistics

- **Total runtime**: 455330s (126.48 CPU-hours)
- **Mean ms/step**: 19.30 ± 14.88

## Figures

- `pilot_c_hill_vs_alpha.png`: Hill exponent versus α by family
- `pilot_c_k_vs_alpha.png`: K versus α by family (power_law cost)
- `pilot_c_floor_hit.png`: Floor-hit rate by status (laplace family)
- `pilot_c_hhi.png`: HHI and top-10 share versus α (laplace family)
- `pilot_c_ccs_vs_alpha.png`: Conglomerate capital share versus α
- `pilot_c_assortativity.png`: Assortativity ratio versus α
- `pilot_c_endo_alpha_hist.png`: Histogram of adopted α (endogenous block)
- `pilot_c_size.png`: K_mean / K* versus α by economy size (M, N)
