# Phase C Pilot Summary

## Overview

- **Scenarios**: 1275
- **Total runtime**: 196424s (54.56 CPU-hours)
- **Mean ms/step**: 14.01 ± 11.57
- **Grid**: 50 × 50 (M × N)

### Block counts

| Block | Count |
|-------|-------|
| correlation | 45 |
| cost-level | 360 |
| endogenous-alpha | 60 |
| equal-split | 180 |
| lookback | 90 |
| main | 540 |

## K versus K* by Family and Cost

Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `K_eff_post_burnin_median`
Reference: `analytics/benchmarks.csv` column `K_star`

### At α = 0.1

| Family | Cost | K [25,75] | K* | K_eff [25,75] | Acc. rate |
|--------|------|-----------|-----|---------------|-----------|
| laplace | exponential | 2.000 [2.000, 2.000] | 15 | 1.831 [1.819, 1.841] | 0.005 |
| laplace | linear | 2.000 [2.000, 2.000] | 10 | 1.829 [1.819, 1.829] | 0.034 |
| laplace | power_law | 2.000 [2.000, 2.000] | 10 | 1.821 [1.819, 1.823] | 0.036 |
| laplace | quadratic | 2.000 [2.000, 2.000] | 12 | 1.835 [1.827, 1.838] | 0.034 |
| normal | exponential | 2.000 [2.000, 2.000] | 12 | 1.837 [1.830, 1.885] | 0.001 |
| normal | linear | 2.000 [2.000, 2.000] | 7 | 1.837 [1.836, 1.844] | 0.037 |
| normal | power_law | 2.000 [2.000, 2.000] | 8 | 1.829 [1.829, 1.830] | 0.042 |
| normal | quadratic | 2.000 [2.000, 2.000] | 10 | 1.848 [1.845, 1.849] | 0.038 |
| t3 | exponential | 2.000 [2.000, 2.000] | 47 | 1.831 [1.817, 1.845] | 0.007 |
| t3 | linear | 2.000 [2.000, 2.000] | 28 | 1.824 [1.816, 1.825] | 0.025 |
| t3 | power_law | 2.000 [2.000, 2.000] | 20 | 1.824 [1.814, 1.827] | 0.029 |
| t3 | quadratic | 2.000 [2.000, 2.000] | 24 | 1.833 [1.831, 1.833] | 0.024 |

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
| 0.01 | 1.063 [1.054, 1.075] |
| 0.02 | 1.065 [1.055, 1.073] |
| 0.05 | 1.062 [1.052, 1.072] |
| 0.1 | 1.064 [1.059, 1.078] |
| 0.2 | 1.064 [1.058, 1.081] |
| 0.3 | 1.072 [1.066, 1.085] |
| 0.4 | 1.070 [1.065, 1.085] |
| 0.5 | 1.081 [1.071, 1.092] |

![Hill vs α](pilot_c_hill_vs_alpha.png)

## Floor-hit Rate by Status

Source: `pilot_c/tidy.csv` columns `floor_hit_rate_standalone`, `floor_hit_rate_member`

### Laplace family

| α | Standalone [25,75] | Member [25,75] |
|---|-------------------|----------------|
| 0.0 | 0.115 [0.109, 0.121] | 0.000 [0.000, 0.000] |
| 0.01 | 0.110 [0.106, 0.115] | 0.004 [0.002, 0.006] |
| 0.02 | 0.105 [0.101, 0.112] | 0.009 [0.006, 0.011] |
| 0.05 | 0.103 [0.096, 0.109] | 0.013 [0.010, 0.014] |
| 0.1 | 0.100 [0.093, 0.106] | 0.015 [0.012, 0.016] |
| 0.2 | 0.096 [0.090, 0.101] | 0.017 [0.015, 0.018] |
| 0.3 | 0.093 [0.087, 0.098] | 0.018 [0.016, 0.019] |
| 0.4 | 0.091 [0.085, 0.096] | 0.017 [0.016, 0.019] |
| 0.5 | 0.091 [0.085, 0.096] | 0.017 [0.015, 0.018] |

![Floor-hit rate](pilot_c_floor_hit.png)

## HHI and Top-10 Share

Source: `pilot_c/tidy.csv` columns `hhi_within_median`, `hhi_aggregate_median`, `top10pct_aggregate_median`

### Laplace family

| α | HHI within [25,75] | HHI agg [25,75] | Top-10 [25,75] | Cong. share [25,75] |
|---|-------------------|-----------------|----------------|---------------------|
| 0.0 | 0.181 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.656 [0.654, 0.661] | 0.000 [0.000, 0.000] |
| 0.01 | 0.181 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.658 [0.654, 0.661] | 0.034 [0.018, 0.068] |
| 0.02 | 0.181 [0.179, 0.184] | 0.004 [0.004, 0.004] | 0.659 [0.657, 0.662] | 0.109 [0.072, 0.133] |
| 0.05 | 0.180 [0.178, 0.184] | 0.004 [0.004, 0.004] | 0.661 [0.657, 0.665] | 0.190 [0.131, 0.199] |
| 0.1 | 0.180 [0.177, 0.183] | 0.004 [0.004, 0.004] | 0.662 [0.659, 0.666] | 0.231 [0.175, 0.235] |
| 0.2 | 0.178 [0.176, 0.182] | 0.004 [0.004, 0.004] | 0.660 [0.657, 0.663] | 0.257 [0.210, 0.269] |
| 0.3 | 0.177 [0.175, 0.181] | 0.004 [0.004, 0.004] | 0.660 [0.654, 0.661] | 0.271 [0.231, 0.276] |
| 0.4 | 0.175 [0.172, 0.179] | 0.004 [0.004, 0.004] | 0.656 [0.653, 0.658] | 0.276 [0.238, 0.285] |
| 0.5 | 0.175 [0.173, 0.179] | 0.004 [0.004, 0.004] | 0.654 [0.651, 0.657] | 0.270 [0.220, 0.276] |

![HHI](pilot_c_hhi.png)

![Cong. Capital Share](pilot_c_ccs_vs_alpha.png)

## Equal-split Check

Comparison of equal-split block to main normal block.
Source: `pilot_c/tidy.csv` column `K_post_burnin_median`, filtered by `block` and `sharing_rule`

### At α = 0.1

| Cost | K (proportional) [25,75] | K (equal) [25,75] |
|------|--------------------------|-------------------|
| exponential | 2.000 [2.000, 2.000] | 2.000 [2.000, 2.000] |
| linear | 2.000 [2.000, 2.000] | 2.000 [2.000, 2.000] |
| power_law | 2.000 [2.000, 2.000] | 2.000 [2.000, 2.000] |
| quadratic | 2.000 [2.000, 2.000] | 2.000 [2.000, 2.000] |

## Lookback Sensitivity

Comparison of lookback block (200, 1000) to main laplace/power_law cell (50).
Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `hill_exponent_median`, filtered by `block` and `lookback`

### At α = 0.1

| Lookback | K [25,75] | Hill [25,75] |
|----------|-----------|--------------|
| 50 (ref) | 2.000 [2.000, 2.000] | 1.072 [1.070, 1.093] |
| 200 | 2.000 [2.000, 2.000] | 1.065 [1.064, 1.066] |
| 1000 | 2.000 [2.000, 2.000] | 1.075 [1.061, 1.075] |

## Correlation Sensitivity

Comparison of correlation block (cross_corr=0.3) to main laplace/power_law cell (cross_corr=0).
Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `hill_exponent_median`, filtered by `block` and `cross_corr`

### At α = 0.1

| Cross-corr | K [25,75] | Hill [25,75] |
|------------|-----------|--------------|
| 0.0 (ref) | 2.000 [2.000, 2.000] | 1.072 [1.070, 1.093] |
| 0.3 | 2.000 [2.000, 2.000] | 1.077 [1.058, 1.077] |

## Endogenous α

Analysis of endogenous-alpha block.
Source: `pilot_c/tidy.csv` columns `alpha_adopted_median`, `alpha_adopted_mean`, `alpha_adopted_std`

### Adopted α by family and cost

| Family | Cost | α adopted (median) | α adopted (mean) | K |
|--------|------|--------------------|------------------|---|
| laplace | exponential | 0.600 | 0.512 | 2.0 |
| laplace | linear | 0.600 | 0.547 | 2.0 |
| laplace | power_law | 0.600 | 0.546 | 2.0 |
| laplace | quadratic | 0.650 | 0.563 | 2.0 |
| normal | exponential | 0.575 | 0.568 | 2.0 |
| normal | linear | 0.600 | 0.517 | 2.0 |
| normal | power_law | 0.600 | 0.527 | 2.0 |
| normal | quadratic | 0.600 | 0.538 | 2.0 |
| t3 | exponential | 0.600 | 0.513 | 2.0 |
| t3 | linear | 0.600 | 0.543 | 2.0 |
| t3 | power_law | 0.600 | 0.531 | 2.0 |
| t3 | quadratic | 0.600 | 0.560 | 2.0 |

![Endogenous α histogram](pilot_c_endo_alpha_hist.png)

## Assortativity

SD of member IQR / SD of random K-subset. Ratio < 1 indicates positive assortment
(conglomerate members have more similar growth volatilities than random groups).

Source: `pilot_c/tidy.csv` column `assortativity_ratio`

### At α = 0.1 (power_law cost)

| Family | Assort. ratio [25,75] |
|--------|----------------------|
| laplace | 0.908 [0.901, 1.006] |
| normal | 0.960 [0.935, 0.963] |
| t3 | 0.931 [0.899, 0.932] |

![Assortativity](pilot_c_assortativity.png)

## Runtime Statistics

- **Total runtime**: 196424s (54.56 CPU-hours)
- **Mean ms/step**: 14.01 ± 11.57

## Figures

- `pilot_c_hill_vs_alpha.png`: Hill exponent versus α by family
- `pilot_c_k_vs_alpha.png`: K versus α by family (power_law cost)
- `pilot_c_floor_hit.png`: Floor-hit rate by status (laplace family)
- `pilot_c_hhi.png`: HHI and top-10 share versus α (laplace family)
- `pilot_c_ccs_vs_alpha.png`: Conglomerate capital share versus α
- `pilot_c_assortativity.png`: Assortativity ratio versus α
- `pilot_c_endo_alpha_hist.png`: Histogram of adopted α (endogenous block)
