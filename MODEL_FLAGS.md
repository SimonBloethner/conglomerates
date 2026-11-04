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

When `alpha_endogenous=True` (requires `g`; `model()` raises otherwise):
1. Each conglomerate carries its own `α_c`, initialized to the run's global `share` parameter (0.1 in the pilot).
2. Every period (C31; before C31 only every `lookback` periods), before pooling, each unit renegotiates its rate.
3. **Demeaned renegotiation (C34).** For the unit's current members over the window {t − l, …, t − 1} (h = min(l, t) before the window is full), the demeaned window is built once with `prepare_window_loggain(members, past_returns, sizes, g)`. For each α′ on the grid {0, 0.05, …, 1} every member's Δ̂_j(α′) is computed with `member_gain_from_prepared(j, K, …, α′, Φ(K), sharing_rule_code)`, the same demeaned log gain the membership and exit tests use.
4. The gain of a move is Δ̂_j(α′) − Δ̂_j(α_C) per member. A candidate under which any member's Δ̂ is −∞ (a replayed p ≤ −1) is infeasible. The unit moves only if the minimum gain over members is > 0 (unanimity) and, among such candidates, takes the one with the largest minimum gain (maximin). Otherwise it keeps α_C. Exact ties between candidates go to the lower α. If Δ̂_j(α_C) = −∞ for some member, every feasible α′ gives that member a gain of +∞ and the minimum is taken over the others.
5. Before C34 the rate was renegotiated on realized (not demeaned) log growth, replayed with `replay_member_growth`. That version was dropped, not kept as a variant: a member with a lucky window could veto a rate that the demeaned comparison shows to benefit everyone (`tests/test_alpha_endogenous.py::test_demeaned_and_level_rules_differ`).
6. Pooling uses `α_c` instead of the global `share` parameter.
7. **Decisions at the unit's α (C31).** Merger and exit tests use the α the unit actually operates at:
   - Δ̂_current of a member is evaluated at `cong_alpha[its unit]` (standalones: Δ̂_current = 0), in the merger kernel and in the exit test.
   - Δ̂_proposed is evaluated at α_prop = `share` for solo+solo, `cong_alpha[unit]` for solo+unit, and for unit+unit the α of the unit with the larger `cong_size` (initiator on ties).
   - When a unit+unit merger executes, the merged unit takes α_prop. A slot newly allocated for solo+solo gets `share` (slots are reused and may hold a stale α).
   - With `alpha_endogenous=False`, every entry of `cong_alpha` equals `share`, so behaviour is unchanged.
   - Under `decision_rule='replay'` the merger and exit tests still use the realized-level replay; the renegotiation is demeaned under either rule. Replay is not combined with endogenous α in the pilot design.

**Unanimity Constraint:** The rule requires all members to benefit from any α change.

**Returns:** `alpha_history[cong, k]` sampled every `metric_every` steps, included in `hyperparameters`.

### Helper Functions

- `find_optimal_alpha(members, past_returns, sizes, current_alpha, Phi_K, sharing_rule_code, g)`: the demeaned renegotiation above. Returns `(alpha, should_change)`.
- `replay_member_growth(...)`: realized-level counterfactual growth of each member under a given α. No production caller since C34; kept as the reference implementation for the replay-rule tests (`tests/test_decision_rule.py`).

## Coordination Cost

### `proportional`
Entry of the `params` list. `proportional=False`, as in every pilot scenario, means the coordination cost is charged as a fraction Φ(K) of unit capital (a growth drag) and is unrelated to `sharing_rule='proportional'`.

### `cost-level` block
The `cost-level` block multiplies `c0` only. That is a level shift of Φ(K) for `power_law` (`c0·K^c1`) and `exponential` (`c0·exp(c1·K)`). For `linear` (`c0 + c1·K`) and `quadratic` (`c0 + c1·K + c2·K²`) it moves only the intercept, so Φ(40) for `linear` goes from 0.00170 to 0.00171 at multiplier 2. The `cost-level-full` block (C33) multiplies `c0`, `c1` and `c2` for these two families, which scales Φ(K) at every K.

## Pooling: limited liability under equal sharing (C31; superseded in part by C34, next section)

Under `sharing_rule='equal'` the pool is shared equally per firm, so a small member can be assigned a pool loss larger than its capital (δ ≤ −1).
- Before C31 such a member was ejected and reset to log size 0. That reset is a gain, not a loss.
- Since C31 the member's δ is capped at −1 + 1e-12 (it loses all its capital and no more). Non-finite δ is treated the same way.
- Its log state drops by log1p(−1 + 1e-12) ≈ −27.6. The floor block then lifts it and, by C27, ejects it, so the event counts as a floor hit and a floor exit.
- `pool_ruin_per_period[step]` counts capped members. It is exported in `hyperparameters`, in `summary['pool_ruin_per_period']` (post-burn-in mean) and as a tidy column.
- Limited liability applies under both sharing rules.
- Under proportional sharing with the pilot's additive management cost, δ = (1−α)·r_i + α·Σ_j w_j r_j − Φ(K) ≥ −1 − Φ(K). The cap can therefore bind only when a member's own shock wipes out essentially all of its capital.
- In the C31 rerun this happened in 23 of the 1,300 runs outside the endogenous-alpha and equal-split blocks, all in the t3 family (main block linear/quadratic/exponential cost, search block power_law). Those 23 runs differ slightly from C30; the other 1,277 are identical.

## Pooling: the shortfall falls on the unit (C34)

C31 capped a ruined member at δ = −1 + 1e-12 and did nothing else. The part of its assigned loss that exceeded its capital was charged to no one; the per-market renormalization then absorbed it, so the member's *market* paid for the unit's shortfall. Since C34 the other members pay it (`process_conglomerate_pooling_numba`, both sharing rules, same code path):

1. Every member's net factor δ_i is computed as before. Its intended post-pooling capital is c_i = s_i(1 + δ_i), and Σ_i c_i = Σ_i s_i(1 + r_i) − Φ(K)·S, the accounting identity the pool conserves (r_i = exp(r̃_i) − 1).
2. A member with δ_i < −1 + 1e-12 (or non-finite δ_i) is capped: it keeps s_i·(1 + CAP), about 1e-12·s_i (0.99998894e-12·s_i in floating point). Its shortfall is s_i(1 + CAP) − c_i, i.e. s_i(−1 − δ_i) plus the ~1e-12·s_i it keeps.
3. The shortfall is charged to the uncapped members in proportion to their post-pooling capital: c_j ← c_j·(1 − SF/C_u), with C_u the uncapped members' capital.
4. Steps 2–3 repeat until no member is below the cap. If the shortfall reaches C_u, every member is capped, the unit holds ~0 and all members hit the floor (a *wipeout*). This should not occur under the pilot's parameters; `run_pilot_c.py` asserts `summary['pool_wipeout_total'] == 0`.
5. After the loop Σ_j s_j,t+1 equals the identity in step 1 to floating-point precision, with capped members contributing their capped size. `tests/test_c34_limited_liability.py` checks |Σ_j s_j,t+1 − [Σ_j s_j(1 + r_j) − Φ·S]| < 1e-9·S for every active unit in every period of a seeded equal-split and a seeded proportional run, and the 2-member, 3-member, cascade, wipeout and proportional-sharing cases by hand.

A unit with no capped member takes the C33 update (log_old + log1p(δ)), so runs without capped members are bit-identical to C33.

Counters (all exported in `hyperparameters`; summary values over t ≥ burn_in unless stated):
- `pool_ruin_per_period`: capped members, including those capped in a later round of a cascade. Summary and tidy column as before.
- `pool_shortfall_per_period` (new): capital charged to other members, in level units (pilot: market mean firm size = 1). Summary `pool_shortfall_per_period` (post-burn-in mean) and tidy column of the same name.
- `pool_wipeout_per_period` (new): units in which every member was capped. Summary `pool_wipeout_total` (whole run).
- `summary['pool_ruin_total']` (new): capped members over the whole run, burn-in included. A run with `pool_ruin_total == 0` never takes the limited-liability branch.

**Decision rule.** `member_gain_from_prepared` still returns −∞ when a replayed p_i ≤ −1, so a proposal, membership or sharing rate under which any member would have been ruined somewhere in the window is rejected outright. The replay does not model the redistribution of a shortfall; this is the conservative side (it rejects arrangements the pooling step would survive).

## Membership timing

- **Lock-in of new members.** A firm is an exit candidate only when `firm_entered < step - lookback`. A new member therefore stays at least `lookback` (l) periods before its first exit test.
- **`firm_entered` reset on unit+unit mergers.** When two units merge, `firm_entered` is set to the merger step for *all* members of the merged unit, not only the newcomers. Every member is locked in again for l periods. On solo+unit merges only the joining firm's `firm_entered` is set.

## Inert Phase-A code paths

These remain in `collaborative_growth.py` but are not exercised by the Phase C pilot (`pilot_design.py`):
- `growth_process='normal_net'` (the function default) and `'lognormal'`: the pilot uses `'log_family'`.
- `renorm_every`: the economy-wide renormalisation is skipped when `market_size_fixed=True` (all pilot scenarios).
- `comparison`, `break_thresh`: unpacked from `params` and stored in `hyperparameters`, but not used by the dynamics.
- `market_corr='random'`: the pilot uses `'identity'`, and a nonzero `cross_corr` replaces the matrix.
- `rho` (common shock): 0 in every pilot scenario.

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

**Pilot:** `pilot_design.py` never sets this flag and `run_pilot_c.py` never passes it, so every pilot scenario, the lookback block included, runs at the default and reviews exits every period.

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
