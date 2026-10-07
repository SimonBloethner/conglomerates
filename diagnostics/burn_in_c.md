# C10: Burn-in Rerun with g=0.02

## Setup

**Parameters**:
- M = N = 50
- g = 0.02 (common time-average growth)
- sigma_range = (0.1, 0.3)
- floor_c = 0.0566 (exponent 1/(1-c) = 1.060)
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
| Hill exponent | 2700 | 0.8907 |
| Mean K | 500 | 0.00 |
| K_eff | 500 | 0.00 |
| Cong capital share | 500 | 0.0000 |
| Floor-hit rate | 5300 | 0.6263 |

### alpha=0.1

| Metric | Convergence Step | Final Mean |
|--------|-----------------|------------|
| Hill exponent | 2700 | 0.8907 |
| Mean K | 7700 | 2.48 |
| K_eff | 5700 | 1.29 |
| Cong capital share | 8000 | 0.0002 |
| Floor-hit rate | 5300 | 0.6258 |

## Burn-in Recommendation

**burn_in = 8000** (max of all convergence steps)

**T = 11000** (burn_in + 3000)

## Hill Exponent vs Theoretical

**Theoretical barrier**: 1/(1-c) = 1/(1-0.0566) = 1.0600

**Measured** (alpha=0, final 2000 steps): 0.8907

**Gap**: 15.97%

## Metric Tables (every 1000 steps)

### alpha=0.0

| Step | Hill | Mean K | K_eff | Cong Capital Share | Floor-hit Rate |
|------|------|--------|-------|-------------------|----------------|
| 1000 | 0.8839 | 0.00 | 0.00 | 0.0000 | 0.6877 |
| 2000 | 0.8885 | 0.00 | 0.00 | 0.0000 | 0.6680 |
| 3000 | 0.8565 | 0.00 | 0.00 | 0.0000 | 0.6764 |
| 4000 | 0.8894 | 0.00 | 0.00 | 0.0000 | 0.6618 |
| 5000 | 0.9111 | 0.00 | 0.00 | 0.0000 | 0.6322 |
| 6000 | 0.9059 | 0.00 | 0.00 | 0.0000 | 0.6311 |
| 7000 | 0.9055 | 0.00 | 0.00 | 0.0000 | 0.6196 |
| 8000 | 0.8672 | 0.00 | 0.00 | 0.0000 | 0.6324 |

### alpha=0.1

| Step | Hill | Mean K | K_eff | Cong Capital Share | Floor-hit Rate |
|------|------|--------|-------|-------------------|----------------|
| 1000 | 0.8912 | 2.66 | 1.37 | 0.0000 | 0.6848 |
| 2000 | 0.8981 | 2.59 | 1.36 | 0.0034 | 0.6659 |
| 3000 | 0.8557 | 2.63 | 1.29 | 0.0051 | 0.6757 |
| 4000 | 0.8765 | 2.41 | 1.27 | 0.0000 | 0.6608 |
| 5000 | 0.9067 | 2.46 | 1.36 | 0.0000 | 0.6292 |
| 6000 | 0.9104 | 2.43 | 1.29 | 0.0000 | 0.6289 |
| 7000 | 0.9010 | 2.56 | 1.31 | 0.0000 | 0.6191 |
| 8000 | 0.8678 | 2.42 | 1.27 | 0.0000 | 0.6311 |

## Figures

- `figures/burnin_c_alpha0.png`: All metrics for alpha=0.0
- `figures/burnin_c_alpha01.png`: All metrics for alpha=0.1
