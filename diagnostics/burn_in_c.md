# C14: Burn-in Rerun with market_size_fixed and c_star

## Setup

**Parameters**:
- M = N = 50
- g = 0.02 (common time-average growth)
- sigma_range = (0.1, 0.3)
- floor_c = 0.12717 (c_star from C13)
- market_size_fixed = True
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
| Hill exponent | 5600 | 1.0705 |
| Mean K | 700 | 2.57 |
| K_eff | 700 | 1.90 |
| Cong capital share | 7900 | 0.2960 |
| Floor-hit rate (standalone) | 4100 | 0.0908 |
| Floor-hit rate (member) | 8000 | 0.0192 |
| Growth gap | 8000 | 0.0236 |

## Burn-in Recommendation

**burn_in = 8000** (max of all convergence steps)

**T = 11000** (burn_in + 3000)

## Acceptance Assertions

### alpha=0

- Hill exponent (final 2000): 1.0644
- Target: 1.06
- Gap: 0.41%
- Status: PASS (within 15%)

### alpha=0.1

- cong_capital_share: 0.2960 > 0.02 -> PASS
- mean K: 2.57 >= 2.5 -> PASS
- K_eff/K: 0.7400 >= 0.5 -> PASS

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
| 1000 | 1.0767 | 2.56 | 1.88 | 0.3191 | 0.0906 | 0.0194 | 0.0224 |
| 2000 | 1.0852 | 2.52 | 1.91 | 0.2979 | 0.0909 | 0.0192 | 0.0240 |
| 3000 | 1.0487 | 2.56 | 1.89 | 0.2849 | 0.0928 | 0.0199 | 0.0231 |
| 4000 | 1.0871 | 2.58 | 1.88 | 0.3029 | 0.0920 | 0.0192 | 0.0255 |
| 5000 | 1.0712 | 2.54 | 1.89 | 0.3053 | 0.0903 | 0.0193 | 0.0224 |
| 6000 | 1.0735 | 2.55 | 1.91 | 0.2792 | 0.0911 | 0.0200 | 0.0205 |
| 7000 | 1.0511 | 2.58 | 1.95 | 0.2768 | 0.0916 | 0.0186 | 0.0276 |
| 8000 | 1.0425 | 2.56 | 1.90 | 0.2872 | 0.0918 | 0.0189 | 0.0268 |

## Figures

- `figures/burnin_c_alpha0.png`: All metrics for alpha=0.0
- `figures/burnin_c_alpha01.png`: All metrics for alpha=0.1

## Notes

Growth gap is computed as the difference in floor-hit rates between standalones and members.
A positive gap indicates standalones hit the floor more often (worse performance).
