# Model Flags Documentation

## Growth Process Flags

### `growth_process`
Specifies how firm growth rates are generated.

Values:
- `"normal_net"` (default): Phase A baseline. `r ~ N(μ, σ²)`, then `log(1+r)` is stored.
- `"lognormal"`: `log δ = (μ - σ²/2) + σ·ε`, so `E[δ] = exp(μ)`.
- `"log_family"`: `log δ = μ + scale·ε` with family-specific innovations.

### `log_family`
Distribution family for innovations when `growth_process == "log_family"`.

Values:
- `"normal"` (default): Gaussian innovations.
- `"laplace"`: Laplace (double exponential) innovations.
- `"student_t"`: Student-t innovations with degrees of freedom specified by `nu`.

Only active when `growth_process == "log_family"`.

### `nu`
Degrees of freedom for Student-t distribution.

Default: `3.0`

Only active when `log_family == "student_t"`.

### `sigma_range`
Tuple `(min_sigma, max_sigma)` specifying bounds for volatility per market.

**Important:** Under `log_family`, `sigma_range` is interpreted as the **IQR (interquartile range)** of `log δ`, not the standard deviation. The scale parameter is computed from IQR using the distribution-specific conversion:
- Normal: `σ = IQR / (2 · Φ⁻¹(0.75)) ≈ IQR / 1.349`
- Laplace: `b = IQR / (2 · ln(2))`
- Student-t: `s = IQR / (2 · t⁻¹(0.75, df=ν))`

## Correlation Flags

### `rho`
Within-market correlation coefficient between firms.

Default: `0.0`

When `rho > 0` (or `rho < 0`), firms within the same market have correlated shocks using a factor model.

### `cross_corr`
Cross-market correlation coefficient.

Default: `0.0`

When `cross_corr > 0`, all pairs of firms across different markets have correlated shocks.

**Gaussian Copula:** When using a non-normal `log_family` with correlation (`rho > 0` or `cross_corr > 0`), a Gaussian copula is used:
1. Draw correlated standard normals using the existing correlation structure
2. Map through Φ (standard normal CDF) to get uniform marginals
3. Map through the family's quantile function

This preserves the marginal distributions while introducing the correlation structure.

## Floor Flags

### `floor_c`
Reflecting floor coefficient at `c × market_median`.

Default: `0.0` (off)

When `floor_c > 0`, after state updates (after pooling, before exit test):
1. For each market, compute the median firm size in levels: `median = median(exp(log_state))`
2. Set log floor: `log_floor = log(c × median)`
3. Any firm below the floor has its log state set to `log_floor`
4. Track `floor_hits[firm]` (per-firm count) and `floor_hits_by_status[step, status]` where status is 0=standalone, 1=member

N is unchanged; no firm is removed. The floor "reflects" firms back to the threshold.

**Stationarity:** With `floor_c > 0`, the size distribution stabilizes. The stationary tail exponent is approximately `1/(1-c)` when using the mean; using the median produces a slightly different value.

**Returns:** `floor_hits` (per-firm array) and `floor_hits_by_status` (steps × 2 array) are included in `hyperparameters`.

## Outcome Metrics Flags

### `metric_every`
Compute outcome metrics every N steps.

Default: `100`

### `burn_in`
Steps to exclude from summary statistics.

Default: `0`

### Computed Metrics

Metrics are computed at steps where `(step + 1) % metric_every == 0` or at the final step. Arrays are indexed by `metric_idx = step // metric_every`.

- `hill_exponent[market, k]`: Hill estimator for tail index on top 10% of firm sizes within each market
- `hhi_within[market, k]`: Herfindahl-Hirschman Index (Σ(share_i)²) within each market
- `hhi_aggregate[k]`: HHI over control units (conglomerates as single units + standalone firms)
- `top10_aggregate[k]`: Top 10% share over control units
- `cong_capital_share[k]`: Capital under conglomerate control / total capital
- `effective_members`: List of (step, [(cong_id, K, K_eff), ...]) tuples, where K_eff = 1/Σ(w_i²) is the effective number of members

### Summary Statistics

Computed over steps `t >= burn_in` and stored in `hyperparameters['summary']`:

- `hill_exponent_median`: Median Hill exponent per market
- `hhi_within_median`: Median within-market HHI per market
- `hhi_aggregate_median`: Median aggregate HHI
- `top10_aggregate_median`: Median top 10% share
- `cong_capital_share_median`: Median conglomerate capital share
- `K_median`: Median conglomerate size K
- `K_eff_over_K_median`: Median K_eff/K ratio
- `floor_hit_rate_standalone`: Floor hit rate for standalone firms
- `floor_hit_rate_member`: Floor hit rate for conglomerate members
- `mergers_per_period`: Mean mergers per period
- `proposals_per_period`: Mean proposals per period
- `exits_per_period`: Mean exits per period
