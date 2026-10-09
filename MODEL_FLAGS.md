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
Reflecting floor coefficient: a firm whose capital falls below `floor_c × (market mean) = (floor_c/N) × total market capital` — i.e. a minimum market share of `floor_c/N` — is reflected to that level. Interpreted as minimum viable scale relative to the market.

Default: `0.0` (off)

When `floor_c > 0`, after state updates (after pooling, before exit test):
1. For each market, compute the mean firm size in levels: `mean = mean(exp(log_state))`
2. Set log floor: `log_floor = log(c × mean)`
3. Any firm below the floor has its log state set to `log_floor`
4. Track `floor_hits[firm]` (per-firm count) and `floor_hits_by_status[step, status]` where status is 0=standalone, 1=member

N is unchanged; no firm is removed. The floor "reflects" firms back to the threshold.

**Stationarity:** With `floor_c > 0`, the size distribution stabilizes following the Levy-Solomon / Gabaix mechanism. The stationary tail exponent is approximately `1/(1-c)` (≈1.053 for c=0.05). Using the market mean (rather than median) ensures the barrier rises with the leaders, producing true stationarity.

**Returns:** `floor_hits` (per-firm array) and `floor_hits_by_status` (steps × 2 array) are included in `hyperparameters`.

## Endogenous Sharing Rate

### `alpha_endogenous`
Enable per-conglomerate endogenous sharing rate adaptation.

Default: `False`

When `alpha_endogenous=True`:
1. Each conglomerate carries its own `α_c` initialized to the run's global `share` parameter
2. Every `lookback` periods, the model evaluates each α on the grid {0, 0.05, ..., 1.0}
3. For each candidate α, it replays the last `lookback` periods of member growth under that α
4. The α that maximizes the minimum member gain (relative to current α) is adopted, but only if `min_gain > 0`
5. Pooling uses `α_c` instead of the global `share` parameter

**Unanimity Constraint:** The rule requires all members to benefit from any α change. With finite samples and K=2 firms, this is rarely satisfied because one firm typically prefers more pooling while the other prefers less.

**Returns:** `alpha_history[cong, k]` sampled every `metric_every` steps, included in `hyperparameters`.

### Helper Functions

- `replay_member_growth(past_states, past_returns, step_states, alpha, management_cost, proportional, sharing_rule_code)`: Compute counterfactual log growth for each member under a given α
- `find_optimal_alpha(...)`: Grid search over α to find the optimal value satisfying unanimity

## Decision Rule

### `decision_rule`
Specifies the rule used to evaluate merger proposals and exit decisions.

Default: `"replay"`

Values:
- `"replay"` (default): Phase B behavior. Replay the last `lookback` periods under the proposed conglomerate and compare log growth. Accept if all members gain.
- `"loggain"`: Demeaned log-growth rule. Removes first-order correlation noise by demeaning returns before computing gains.

**Loggain Rule:**

The `loggain` rule addresses a fundamental challenge: when firms share correlated shocks (e.g., a common market factor), the replay comparison is dominated by that common noise rather than the diversification benefit.

*Demeaning procedure:*
1. For each period τ in the lookback window:
   - Compute pooled return: `p_iτ = Σ_j w_j · r̃_jτ - mc_i` where `w_j` are shares and `mc_i` is management cost
   - Compute demeaned outside: `ε_iτ = r̃_iτ - mean_τ(r̃_jτ) + g` where mean is over conglomerate members
2. Gain: `Δ̂_i = (1/h) Σ_τ [ log(1 + p_iτ) - log(1 + ε_iτ) ]`
3. Merger accepted if `Δ̂_i > max(0, Δ̂_current)` for all members
4. Exit if `Δ̂_current < 0`

**Requires:** `g` must be specified when `decision_rule == "loggain"` (growth rate for demeaning).

**Phase B identity:** With `decision_rule == "replay"` (default), behavior is byte-for-byte identical to Phase B.

### `exit_review_every`
Periodic exit review interval for loggain decision rule.

Default: `1` (every step)

**C21 convention:** Main results use `exit_review_every = 5` (one-tenth of the default `lookback = 50`). The lookback block uses `exit_review_every = max(1, lookback // 10)` to maintain proportionality.

When `exit_review_every > 1`:
- Conglomerate members run the exit test only at steps where `(step - cong_created_step) % exit_review_every == 0`
- The review schedule resets when a conglomerate's composition changes (merger or new member)
- Models "boards review the arrangement every n periods"

**Only affects loggain:** Under `decision_rule == "replay"` (default), this flag is ignored and exits are evaluated every step.

## Fixed Market Size

### `market_size_fixed`
Fixed market size mode for relative dynamics.

Default: `False`

When `market_size_fixed=True` (requires `growth_process='log_family'`):

**Relative Returns:**
1. Draw raw log shocks `x_{m,j}` as usual (family, IQR per market, g as location)
2. Compute market growth factor: `G_m = Σ_j s_{m,j}·exp(x_{m,j}) / Σ_j s_{m,j}` (sizes at start of step)
3. Compute relative return: `r̃_{m,j} = x_{m,j} - log(G_m)`
4. Store `r̃` (not `x`) in `firm_log_returns_buffer` and `firm_outside_log_profits`
5. Use `r̃` everywhere `x` was used (state update, pooling, replay in merger kernel, exit test)

By construction, a market of standalone firms keeps its total capital exactly constant.

**Per-Market Renormalization:**
After pooling and floor reflection, the model renormalizes each market so its mean size equals 1:
- Subtract `log(Σ_j s_{m,j} / N)` from every firm in market m (all rows of circular state buffer)
- This absorbs small net capital flows from cross-market pooling and floor reflection
- Records `|log(Σs/N)|` per market per period; returns mean as `renorm_correction_mean`

**Interpretation:**
- With fixed market size, every market has mean size 1 at every step
- The floor `floor_c` represents a minimum market share of `floor_c/N`
- Cross-market divergence is eliminated; markets differ only in relative dynamics
- The g parameter has no effect on levels (already normalized), but is kept as a stored parameter

**Skipped:**
The C8 economy-wide renormalization is redundant when this flag is on and is skipped.

**Returns:**
- `renorm_correction_mean` in `summary`: mean correction per step (should be small, <1e-3)
- `renorm_corrections` in `hyperparameters`: per-step correction array

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
- `top10pct_aggregate[k]`: Top 10% share over control units
- `cong_capital_share[k]`: Capital under conglomerate control / total capital
- `effective_members`: List of (step, [(cong_id, K, K_eff), ...]) tuples, where K_eff = 1/Σ(w_i²) is the effective number of members

### Summary Statistics

Computed over steps `t >= burn_in` and stored in `hyperparameters['summary']`:

- `hill_exponent_median`: Median Hill exponent per market
- `hhi_within_median`: Median within-market HHI per market
- `hhi_aggregate_median`: Median aggregate HHI
- `top10pct_aggregate_median`: Median top 10% share
- `cong_capital_share_median`: Median conglomerate capital share
- `K_median`: Median conglomerate size K
- `K_eff_over_K_median`: Median K_eff/K ratio
- `floor_hit_rate_standalone`: Floor hit rate for standalone firms
- `floor_hit_rate_member`: Floor hit rate for conglomerate members
- `mergers_per_period`: Mean mergers per period
- `proposals_per_period`: Mean proposals per period
- `exits_per_period`: Mean exits per period

## Event Study

The event study module implements a matched difference-in-differences (DiD) design to measure the causal effect of conglomerate membership on firm growth.

### Event Recording

**Entry events:** Recorded at every step where a firm joins a conglomerate, subject to:
- Boundary condition: `lookback ≤ step ≤ steps - lookback` (ensures full pre/post windows)
- All entries are recorded, not just first entries

**Exit events:** Recorded whenever a firm leaves a conglomerate (voluntary exit or dissolution).

### Event Selection

At post-processing, each entry event is evaluated:
1. **Minimum tenure:** The firm must stay in the conglomerate for at least `lookback` periods after entry
2. **Re-entry handling:** If a firm enters, exits early (< lookback periods), and re-enters, the early entry is discarded; only entries with sufficient tenure contribute events
3. **Post-burn-in:** Events with `entry_step ≥ burn_in` are flagged via `event_did_post_median`

### Control Selection

For each treated firm at entry step t:
1. Identify all firms in the same market
2. Select controls that were **standalone throughout** the window [t - lookback, t + lookback]
3. Compute the market-level control mean

### Output Units

All DiD estimates are reported as **mean log share change per period**:
- `event_did_treated = (1/l) × Σ [log_state(t+τ) - log_state(t-l+τ)]` for τ ∈ [0, l)
- `event_did_control`: Same computation averaged over control firms
- `event_did_diff = event_did_treated - event_did_control`

This normalization allows comparison across different lookback values.

### Output Fields

Stored in `hyperparameters`:
- `event_n_events`: Total number of valid entry events (with ≥l tenure and matched controls)
- `event_n_matched`: Number of events with at least one valid control firm
- `event_did_treated[k]`: Treated firm's mean log growth per period
- `event_did_control[k]`: Control mean log growth per period
- `event_did_diff[k]`: Difference (treatment effect per period)
- `event_entry_step[k]`: Step when the treated firm entered
- `event_did_post_median[k]`: 1 if entry_step ≥ burn_in, else 0
