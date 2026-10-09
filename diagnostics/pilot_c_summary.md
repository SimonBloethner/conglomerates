# Phase C Pilot Summary

## Overview

- **Scenarios**: 1370
- **Total runtime**: 419367s (116.49 CPU-hours)
- **Mean ms/step**: 27.83 ± 29.87
- **Grid**: 50 × 50 (M × N)

### Block counts

| Block | Count |
|-------|-------|
| correlation | 45 |
| cost-level | 360 |
| endogenous-alpha | 60 |
| equal-split | 180 |
| floor-level | 90 |
| lookback | 50 |
| main | 540 |
| rule-replay | 45 |

## K versus K* by Family and Cost

Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `K_eff_post_burnin_median`
Reference: `analytics/benchmarks.csv` column `K_star`

### At α = 0.1

| Family | Cost | K [25,75] | K* | K_eff [25,75] | Acc. rate |
|--------|------|-----------|-----|---------------|-----------|
| laplace | exponential | 6.000 [6.000, 6.000] | 15 | 3.397 [3.357, 3.472] | 0.055 |
| laplace | linear | 6.000 [6.000, 7.000] | 10 | 3.329 [3.255, 3.403] | 0.035 |
| laplace | power_law | 7.000 [6.000, 7.000] | 10 | 3.466 [3.464, 3.490] | 0.031 |
| laplace | quadratic | 8.000 [8.000, 9.000] | 12 | 3.858 [3.780, 3.890] | 0.035 |
| normal | exponential | 4.000 [4.000, 5.000] | 12 | 2.940 [2.869, 2.995] | 0.058 |
| normal | linear | 4.000 [4.000, 4.000] | 7 | 2.891 [2.823, 2.920] | 0.051 |
| normal | power_law | 5.000 [5.000, 5.000] | 8 | 3.062 [3.058, 3.167] | 0.042 |
| normal | quadratic | 7.000 [7.000, 7.000] | 10 | 3.781 [3.745, 3.831] | 0.043 |
| t3 | exponential | 6.000 [6.000, 6.000] | 47 | 3.295 [3.273, 3.308] | 0.051 |
| t3 | linear | 7.000 [6.000, 7.000] | 28 | 3.403 [3.171, 3.490] | 0.031 |
| t3 | power_law | 7.000 [7.000, 7.000] | 20 | 3.379 [3.375, 3.394] | 0.030 |
| t3 | quadratic | 9.000 [9.000, 9.000] | 24 | 3.696 [3.677, 3.764] | 0.032 |

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
| 0.01 | 1.059 [1.051, 1.071] |
| 0.02 | 1.061 [1.052, 1.071] |
| 0.05 | 1.064 [1.053, 1.072] |
| 0.1 | 1.068 [1.062, 1.078] |
| 0.2 | 1.107 [1.099, 1.113] |
| 0.3 | 1.154 [1.145, 1.160] |
| 0.4 | 1.215 [1.206, 1.227] |
| 0.5 | 1.283 [1.262, 1.301] |

![Hill vs α](pilot_c_hill_vs_alpha.png)

## Floor-hit Rate by Status

Source: `pilot_c/tidy.csv` columns `floor_hit_rate_standalone`, `floor_hit_rate_member`

### Laplace family

| α | Standalone [25,75] | Member [25,75] |
|---|-------------------|----------------|
| 0.0 | 0.115 [0.109, 0.121] | 0.000 [0.000, 0.000] |
| 0.01 | 0.030 [0.021, 0.053] | 0.082 [0.052, 0.092] |
| 0.02 | 0.013 [0.010, 0.038] | 0.096 [0.068, 0.105] |
| 0.05 | 0.006 [0.005, 0.016] | 0.099 [0.091, 0.107] |
| 0.1 | 0.004 [0.003, 0.007] | 0.097 [0.091, 0.103] |
| 0.2 | 0.003 [0.002, 0.004] | 0.086 [0.082, 0.093] |
| 0.3 | 0.002 [0.002, 0.002] | 0.075 [0.071, 0.080] |
| 0.4 | 0.001 [0.001, 0.001] | 0.064 [0.060, 0.067] |
| 0.5 | 0.000 [0.000, 0.000] | 0.052 [0.048, 0.055] |

![Floor-hit rate](pilot_c_floor_hit.png)

## HHI and Top-10 Share

Source: `pilot_c/tidy.csv` columns `hhi_within_median`, `hhi_aggregate_median`, `top10pct_aggregate_median`

### Laplace family

| α | HHI within [25,75] | HHI agg [25,75] | Top-10 [25,75] | Cong. share [25,75] |
|---|-------------------|-----------------|----------------|---------------------|
| 0.0 | 0.181 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.656 [0.654, 0.661] | 0.000 [0.000, 0.000] |
| 0.01 | 0.181 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.581 [0.562, 0.613] | 0.321 [0.210, 0.386] |
| 0.02 | 0.181 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.531 [0.504, 0.581] | 0.457 [0.316, 0.497] |
| 0.05 | 0.182 [0.180, 0.184] | 0.005 [0.004, 0.005] | 0.471 [0.451, 0.521] | 0.580 [0.473, 0.611] |
| 0.1 | 0.179 [0.176, 0.182] | 0.005 [0.005, 0.005] | 0.432 [0.416, 0.464] | 0.660 [0.579, 0.692] |
| 0.2 | 0.169 [0.166, 0.173] | 0.006 [0.006, 0.007] | 0.397 [0.385, 0.417] | 0.736 [0.694, 0.781] |
| 0.3 | 0.156 [0.153, 0.159] | 0.008 [0.008, 0.009] | 0.371 [0.360, 0.396] | 0.827 [0.802, 0.867] |
| 0.4 | 0.142 [0.139, 0.143] | 0.010 [0.010, 0.011] | 0.339 [0.332, 0.349] | 0.931 [0.896, 0.952] |
| 0.5 | 0.126 [0.123, 0.127] | 0.011 [0.011, 0.012] | 0.305 [0.298, 0.313] | 0.984 [0.973, 0.989] |

![HHI](pilot_c_hhi.png)

![Cong. Capital Share](pilot_c_ccs_vs_alpha.png)

## Equal-split Check

Comparison of equal-split block to main normal block.
Source: `pilot_c/tidy.csv` column `K_post_burnin_median`, filtered by `block` and `sharing_rule`

### At α = 0.1

| Cost | K (proportional) [25,75] | K (equal) [25,75] |
|------|--------------------------|-------------------|
| exponential | 4.000 [4.000, 5.000] | 2.000 [2.000, 2.000] |
| linear | 4.000 [4.000, 4.000] | 2.000 [2.000, 3.000] |
| power_law | 5.000 [5.000, 5.000] | 3.000 [3.000, 3.000] |
| quadratic | 7.000 [7.000, 7.000] | 3.000 [3.000, 3.000] |

## Lookback Sensitivity

Comparison of lookback block (200, 1000) to main laplace/power_law cell (50).
Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `hill_exponent_median`, filtered by `block` and `lookback`

### At α = 0.1

| Lookback | K [25,75] | Hill [25,75] |
|----------|-----------|--------------|
| 50 (ref) | 7.000 [6.000, 7.000] | 1.077 [1.074, 1.093] |
| 20 | 4.000 [4.000, 4.000] | 1.065 [1.061, 1.066] |
| 50 | 7.000 [6.000, 7.000] | 1.070 [1.065, 1.083] |
| 100 | 9.000 [9.000, 9.000] | 1.077 [1.074, 1.087] |
| 200 | 9.000 [8.000, 9.000] | 1.086 [1.084, 1.086] |
| 500 | 9.000 [9.000, 9.000] | 1.090 [1.083, 1.093] |

## Correlation Sensitivity

Comparison of correlation block (cross_corr=0.3) to main laplace/power_law cell (cross_corr=0).
Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `hill_exponent_median`, filtered by `block` and `cross_corr`

### At α = 0.1

| Cross-corr | K [25,75] | Hill [25,75] |
|------------|-----------|--------------|
| 0.0 (ref) | 7.000 [6.000, 7.000] | 1.077 [1.074, 1.093] |
| 0.3 | 7.000 [6.000, 7.000] | 1.074 [1.065, 1.077] |

## Endogenous α

Analysis of endogenous-alpha block.
Source: `pilot_c/tidy.csv` columns `alpha_adopted_median`, `alpha_adopted_mean`, `alpha_adopted_std`

### Adopted α by family and cost

| Family | Cost | α adopted (median) | α adopted (mean) | K |
|--------|------|--------------------|------------------|---|
| laplace | exponential | 0.600 | 0.553 | 14.0 |
| laplace | linear | 0.600 | 0.530 | 9.0 |
| laplace | power_law | 0.550 | 0.508 | 9.0 |
| laplace | quadratic | 0.475 | 0.452 | 11.0 |
| normal | exponential | 0.800 | 0.720 | 15.0 |
| normal | linear | 0.600 | 0.561 | 7.0 |
| normal | power_law | 0.550 | 0.513 | 8.0 |
| normal | quadratic | 0.450 | 0.436 | 10.0 |
| t3 | exponential | 0.600 | 0.571 | 14.0 |
| t3 | linear | 0.600 | 0.537 | 9.0 |
| t3 | power_law | 0.600 | 0.533 | 10.0 |
| t3 | quadratic | 0.550 | 0.498 | 11.0 |

![Endogenous α histogram](pilot_c_endo_alpha_hist.png)

## Assortativity

SD of member IQR / SD of random K-subset. Ratio < 1 indicates positive assortment
(conglomerate members have more similar growth volatilities than random groups).

Source: `pilot_c/tidy.csv` column `assort_iqr`

### At α = 0.1 (power_law cost)

| Family | Assort. ratio [25,75] |
|--------|----------------------|
| laplace | — |
| normal | — |
| t3 | — |

![Assortativity](pilot_c_assortativity.png)

## Runtime Statistics

- **Total runtime**: 419367s (116.49 CPU-hours)
- **Mean ms/step**: 27.83 ± 29.87

## Figures

- `pilot_c_hill_vs_alpha.png`: Hill exponent versus α by family
- `pilot_c_k_vs_alpha.png`: K versus α by family (power_law cost)
- `pilot_c_floor_hit.png`: Floor-hit rate by status (laplace family)
- `pilot_c_hhi.png`: HHI and top-10 share versus α (laplace family)
- `pilot_c_ccs_vs_alpha.png`: Conglomerate capital share versus α
- `pilot_c_assortativity.png`: Assortativity ratio versus α
- `pilot_c_endo_alpha_hist.png`: Histogram of adopted α (endogenous block)
