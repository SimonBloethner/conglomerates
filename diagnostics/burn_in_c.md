# C18: Burn-in with decision_rule=loggain

## Setup

**Parameters**:
- M = N = 50
- g = 0.02 (common time-average growth)
- sigma_range = (0.1, 0.3)
- floor_c = 0.12717 (c_star from C13)
- market_size_fixed = True
- decision_rule = loggain (C18)
- T = 8000 for burn-in analysis
- 5 replications with seeds 42 to 46
- metric_every = 100
- Sharing rule: proportional
- Growth process: log_family (laplace)
- Cost function: power_law

## Convergence Results

Using 500-step rolling mean vs final 2000-step mean, 2% tolerance.

### alpha=0.0

| Metric | Convergence Step | Final Mean |
|--------|-----------------|------------|
| Hill exponent | 5600 | 1.0644 |
| Mean K | 500 | 0.00 |
| K_eff | 500 | 0.00 |
| Cong capital share | 500 | 0.0000 |
| Floor-hit rate (standalone) | 600 | 0.1117 |
| Floor-hit rate (member) | 500 | 0.0000 |
| Growth gap | 600 | 0.1117 |

### alpha=0.1

| Metric | Convergence Step | Final Mean |
|--------|-----------------|------------|
| Hill exponent | 2600 | 1.0806 |
| Mean K | 3200 | 8.03 |
| K_eff | 2500 | 4.03 |
| Cong capital share | 7200 | 0.7990 |
| Floor-hit rate (standalone) | 8000 | 0.0014 |
| Floor-hit rate (member) | 2000 | 0.0985 |
| Growth gap | 8000 | -0.0656 |

## Burn-in Recommendation

**burn_in = 8000** (max of all convergence steps)

**T = 11000** (burn_in + 3000)

## Acceptance Assertions

### alpha=0

- Hill exponent (final 2000): 1.0644
- Target: 1.06
- Gap: 0.41%
- Status: PASS (within 15%)

### alpha=0.1 (C18 loggain thresholds)

- mean K: 8.03 >= 3.0 -> PASS
- cong_capital_share: 0.7990 > 0.05 -> PASS
- mergers_per_period: 0.78 < 2.0 -> PASS
- K_eff/K: 0.5021 >= 0.5 -> PASS

## Metric Tables (every 1000 steps)

### alpha=0.0

| Step | Hill | Mean K | K_eff | Cong Share | FHR Stand | FHR Memb | Gap |
|------|------|--------|-------|------------|-----------|----------|-----|
| 1000 | 1.0779 | 0.00 | 0.00 | 0.0000 | 0.1117 | 0.0000 | 0.1117 |
| 2000 | 1.0801 | 0.00 | 0.00 | 0.0000 | 0.1119 | 0.0000 | 0.1119 |
| 3000 | 1.0410 | 0.00 | 0.00 | 0.0000 | 0.1143 | 0.0000 | 0.1143 |
| 4000 | 1.0968 | 0.00 | 0.00 | 0.0000 | 0.1130 | 0.0000 | 0.1130 |
| 5000 | 1.0741 | 0.00 | 0.00 | 0.0000 | 0.1113 | 0.0000 | 0.1113 |
| 6000 | 1.0767 | 0.00 | 0.00 | 0.0000 | 0.1127 | 0.0000 | 0.1127 |
| 7000 | 1.0410 | 0.00 | 0.00 | 0.0000 | 0.1119 | 0.0000 | 0.1119 |
| 8000 | 1.0400 | 0.00 | 0.00 | 0.0000 | 0.1122 | 0.0000 | 0.1122 |

### alpha=0.1

| Step | Hill | Mean K | K_eff | Cong Share | FHR Stand | FHR Memb | Gap |
|------|------|--------|-------|------------|-----------|----------|-----|
| 1000 | 1.0860 | 7.11 | 3.75 | 0.8071 | 0.0012 | 0.0994 | -0.0722 |
| 2000 | 1.0779 | 7.63 | 3.90 | 0.7923 | 0.0015 | 0.0989 | -0.0626 |
| 3000 | 1.0753 | 7.88 | 4.02 | 0.7734 | 0.0013 | 0.1017 | -0.0735 |
| 4000 | 1.0774 | 7.94 | 4.03 | 0.7819 | 0.0015 | 0.1003 | -0.0649 |
| 5000 | 1.1081 | 8.04 | 4.02 | 0.8075 | 0.0015 | 0.0979 | -0.0625 |
| 6000 | 1.0887 | 8.09 | 4.04 | 0.7962 | 0.0017 | 0.0987 | -0.0581 |
| 7000 | 1.0727 | 8.13 | 4.15 | 0.7784 | 0.0012 | 0.0983 | -0.0702 |
| 8000 | 1.0333 | 7.98 | 4.08 | 0.7989 | 0.0014 | 0.0992 | -0.0672 |

## Figures

- `figures/burnin_c_alpha0.png`: All metrics for alpha=0.0
- `figures/burnin_c_alpha01.png`: All metrics for alpha=0.1

## Notes

Growth gap is computed as the difference in floor-hit rates between standalones and members.
A positive gap indicates standalones hit the floor more often (worse performance).
