# Phase C Pilot Summary

## Overview

- **Scenarios**: 1275
- **Total runtime**: 91441s (25.40 CPU-hours)
- **Mean ms/step**: 6.52 ± 2.07
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
| laplace | exponential | 2.000 [2.000, 2.500] | 15 | 1.125 [1.079, 1.330] | 0.000 |
| laplace | linear | 2.000 [2.000, 2.000] | 10 | 1.201 [1.195, 1.208] | 0.004 |
| laplace | power_law | 2.000 [2.000, 2.000] | 10 | 1.134 [1.121, 1.145] | 0.004 |
| laplace | quadratic | 2.000 [2.000, 2.000] | 12 | 1.213 [1.180, 1.217] | 0.003 |
| normal | exponential | 2.000 [2.000, 2.000] | 12 | 1.327 [1.316, 1.519] | 0.000 |
| normal | linear | 2.000 [2.000, 2.000] | 7 | 1.145 [1.143, 1.203] | 0.007 |
| normal | power_law | 2.000 [2.000, 2.000] | 8 | 1.098 [1.093, 1.130] | 0.009 |
| normal | quadratic | 2.000 [2.000, 2.000] | 10 | 1.174 [1.171, 1.206] | 0.006 |
| t3 | exponential | 2.000 [2.000, 2.000] | 47 | 1.410 [1.287, 1.763] | 0.000 |
| t3 | linear | 2.000 [2.000, 2.000] | 28 | 1.194 [1.185, 1.292] | 0.002 |
| t3 | power_law | 2.000 [2.000, 2.000] | 20 | 1.147 [1.138, 1.155] | 0.003 |
| t3 | quadratic | 2.000 [2.000, 2.000] | 24 | 1.212 [1.195, 1.226] | 0.002 |

## Hill Exponent

Source: `pilot_c/tidy.csv` column `hill_exponent_median`

**Theoretical at α=0**: 1/(1−c) = 1/(1−0.0566) = 1.0600

### At α = 0

| Family | Hill median [25,75] |
|--------|---------------------|
| laplace | 0.892 [0.883, 0.902] |
| normal | 0.903 [0.896, 0.911] |
| t3 | 0.849 [0.840, 0.855] |

### Hill exponent versus α (laplace family)

| α | Hill median [25,75] |
|---|---------------------|
| 0.0 | 0.892 [0.883, 0.902] |
| 0.01 | 0.892 [0.883, 0.902] |
| 0.02 | 0.892 [0.883, 0.902] |
| 0.05 | 0.890 [0.883, 0.901] |
| 0.1 | 0.891 [0.884, 0.906] |
| 0.2 | 0.895 [0.885, 0.903] |
| 0.3 | 0.895 [0.885, 0.900] |
| 0.4 | 0.894 [0.886, 0.904] |
| 0.5 | 0.890 [0.884, 0.903] |

![Hill vs α](pilot_c_hill_vs_alpha.png)

## Floor-hit Rate by Status

Source: `pilot_c/tidy.csv` columns `floor_hit_rate_standalone`, `floor_hit_rate_member`

### Laplace family

| α | Standalone [25,75] | Member [25,75] |
|---|-------------------|----------------|
| 0.0 | 0.643 [0.602, 0.691] | 0.000 [0.000, 0.000] |
| 0.01 | 0.642 [0.601, 0.691] | 0.000 [0.000, 0.001] |
| 0.02 | 0.642 [0.600, 0.690] | 0.001 [0.001, 0.002] |
| 0.05 | 0.641 [0.599, 0.689] | 0.003 [0.001, 0.004] |
| 0.1 | 0.640 [0.596, 0.687] | 0.004 [0.003, 0.005] |
| 0.2 | 0.638 [0.596, 0.685] | 0.006 [0.003, 0.006] |
| 0.3 | 0.638 [0.594, 0.685] | 0.006 [0.004, 0.007] |
| 0.4 | 0.638 [0.595, 0.684] | 0.006 [0.004, 0.007] |
| 0.5 | 0.638 [0.595, 0.683] | 0.006 [0.004, 0.007] |

![Floor-hit rate](pilot_c_floor_hit.png)

## HHI and Top-10 Share

Source: `pilot_c/tidy.csv` columns `hhi_within_median`, `hhi_aggregate_median`, `top10_aggregate_median`

### Laplace family

| α | HHI within [25,75] | HHI agg [25,75] | Top-10 [25,75] | Cong. share [25,75] |
|---|-------------------|-----------------|----------------|---------------------|
| 0.0 | 0.268 [0.265, 0.273] | 0.204 [0.188, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| 0.01 | 0.268 [0.265, 0.273] | 0.205 [0.188, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| 0.02 | 0.268 [0.265, 0.273] | 0.204 [0.187, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| 0.05 | 0.268 [0.266, 0.272] | 0.204 [0.188, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| 0.1 | 0.268 [0.264, 0.273] | 0.203 [0.189, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| 0.2 | 0.268 [0.266, 0.271] | 0.201 [0.191, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| 0.3 | 0.268 [0.266, 0.272] | 0.200 [0.187, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| 0.4 | 0.268 [0.264, 0.271] | 0.204 [0.187, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| 0.5 | 0.269 [0.267, 0.272] | 0.201 [0.191, 0.234] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

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
| 50 (ref) | 2.000 [2.000, 2.000] | 0.899 [0.882, 0.909] |
| 200 | 2.000 [2.000, 2.000] | 0.888 [0.885, 0.892] |
| 1000 | 2.000 [2.000, 2.000] | 0.895 [0.889, 0.897] |

## Correlation Sensitivity

Comparison of correlation block (cross_corr=0.3) to main laplace/power_law cell (cross_corr=0).
Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `hill_exponent_median`, filtered by `block` and `cross_corr`

### At α = 0.1

| Cross-corr | K [25,75] | Hill [25,75] |
|------------|-----------|--------------|
| 0.0 (ref) | 2.000 [2.000, 2.000] | 0.899 [0.882, 0.909] |
| 0.3 | 2.000 [2.000, 2.000] | 0.893 [0.891, 0.898] |

## Endogenous α

Analysis of endogenous-alpha block.
Source: `pilot_c/tidy.csv` columns `alpha_adopted_median`, `alpha_adopted_mean`, `alpha_adopted_std`

### Adopted α by family and cost

| Family | Cost | α adopted (median) | α adopted (mean) | K |
|--------|------|--------------------|------------------|---|
| laplace | exponential | 0.300 | 0.297 | 2.0 |
| laplace | linear | 0.450 | 0.433 | 2.0 |
| laplace | power_law | 0.450 | 0.430 | 2.0 |
| laplace | quadratic | 0.450 | 0.426 | 2.0 |
| normal | exponential | 0.325 | 0.315 | 2.0 |
| normal | linear | 0.450 | 0.408 | 2.0 |
| normal | power_law | 0.500 | 0.434 | 2.0 |
| normal | quadratic | 0.450 | 0.412 | 2.0 |
| t3 | exponential | 0.300 | 0.317 | 2.0 |
| t3 | linear | 0.450 | 0.409 | 2.0 |
| t3 | power_law | 0.450 | 0.415 | 2.0 |
| t3 | quadratic | 0.400 | 0.400 | 2.0 |

![Endogenous α histogram](pilot_c_endo_alpha_hist.png)

## Assortativity

SD of member IQR / SD of random K-subset. Ratio < 1 indicates positive assortment
(conglomerate members have more similar growth volatilities than random groups).

Source: `pilot_c/tidy.csv` column `assortativity_ratio`

### At α = 0.1 (power_law cost)

| Family | Assort. ratio [25,75] |
|--------|----------------------|
| laplace | — |
| normal | — |
| t3 | — |

![Assortativity](pilot_c_assortativity.png)

## Runtime Statistics

- **Total runtime**: 91441s (25.40 CPU-hours)
- **Mean ms/step**: 6.52 ± 2.07

## Figures

- `pilot_c_hill_vs_alpha.png`: Hill exponent versus α by family
- `pilot_c_k_vs_alpha.png`: K versus α by family (power_law cost)
- `pilot_c_floor_hit.png`: Floor-hit rate by status (laplace family)
- `pilot_c_hhi.png`: HHI and top-10 share versus α (laplace family)
- `pilot_c_ccs_vs_alpha.png`: Conglomerate capital share versus α
- `pilot_c_assortativity.png`: Assortativity ratio versus α
- `pilot_c_endo_alpha_hist.png`: Histogram of adopted α (endogenous block)
