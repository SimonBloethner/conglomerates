# Referee Fixes Mapping

This document maps Reviewer #1 items to the commits that address them on the `referee-fixes` branch.

## Summary Table

| Reviewer Item | Description | Commit / Status | Section |
|---------------|-------------|-----------------|---------|
| Analytical 1 | Growth formula: g = μ − σ²/2 vs E[log(1+r)] | **Phase B §1** | Growth process |
| Analytical 2 | log Δ with Δ possibly negative | **Phase B §1** | Growth process |
| Merger 1 | Two-firm matches untested | `d76ea51` | §2d |
| Merger 2 | Pool-level comparison (should be per-member) | `d76ea51` | §2d |
| Merger 3 | α=0 switch (blocked mergers) | `c91e06e` | §2c |
| Calibration 1 | Midpoint cost grids | `f41e29b` | §2e |
| Calibration 2 | Transposed shock-to-firm mapping | `0b31115` | §2a |
| Calibration 3 | Random correlation matrix (unseeded) | `1af30a7` | §1b/§2b |
| Calibration 4 | α=0.02 results contradict text | **Resolved by rerun** | — |
| Minor 1 | Table 2 inference (OLS on time series) | **Phase B §5** | Inference |
| Minor 2 | Unseeded Numba RNG | `70f8407` | §1a |
| (Additional) | Numba-seed bug | `70f8407` | §1a |
| (Additional) | Non-deterministic hash() seed | `TBD` | §0a |

## Detailed Commit Log

### §0 Environment: np.trapezoid and cache=True (`c8f6f77`)
- Replaced deprecated `np.trapz` with `np.trapezoid` (NumPy 2.0+ compatibility)
- Added `cache=True` to `@nb.njit` decorators for faster startup

### §0a Non-deterministic seed fix (`TBD`)
- Replaced `hash((scenario_name, exp_id))` with `zlib.crc32(f"{scenario_name}:{exp_id}".encode())`
- Python's `hash()` is salted per-process (PYTHONHASHSEED), causing different seeds across SLURM tasks
- Added `tests/test_seed_determinism.py` to verify deterministic seeding

### §0c Merger kernel optimization (`TBD`)
- Precompute `sum_Delta_tau` and `sum_s_tau` once per τ before firm loop
- Reduces complexity from O(K²·h) to O(K·h) where K = merged size, h = lookback
- Output is bit-identical to original implementation

### §1a Reproducibility: Seed Numba RNG (`70f8407`)
- Added `seed_numba(s)` function that calls `np.random.seed(s)` inside `@njit`
- Numba maintains separate per-thread RNG state that needs explicit seeding
- Called after standard `np.random.seed()` in experiment runner
- **Addresses Minor 2 and Numba-seed bug**

### §1b/§2b Market correlation (`1af30a7`)
- Changed default correlation matrix from random to identity
- Added `--market_corr` CLI argument: "identity" (default) or "random"
- When "random", correlation matrix seeded from same seed for reproducibility
- **Addresses Calibration 3**

### §2a Fix transposed shock-to-firm mapping (`0b31115`)
- Changed `.T.flatten()` to `.ravel()` for row-major order
- Shocks now correctly map to firms within each market
- **Addresses Calibration 2**

### §2c Remove α=0 merger suppression (`c91e06e`)
- Removed the `if share > 0:` check that blocked mergers when α=0
- Added `proposals_per_period` counter to track merger activity
- Enables proper counterfactual comparison at α=0 baseline
- **Addresses Merger 3**

### §2d Per-member merger acceptance test (`d76ea51`)
- Changed merger acceptance from conglomerate-level to per-member test
- Each member's synthetic log-growth must exceed realized log-growth
- Reflects that each firm must consent to the merger
- **Addresses Merger 1 and Merger 2**

### §2e Cost calibration (`f41e29b`)
- Changed from absolute parameter grids to multiplicative perturbations
- Level multipliers: [0.5, 1.0, 2.0] applied to c0
- Shape multipliers: [0.8, 1.0, 1.25] applied to c1/c2
- Generated `robustness_cost_table.md` with 84 scenarios
- **Addresses Calibration 1**

### §2f Common random numbers (`732ddcd`)
- Changed seed scheme: `hash((scenario_name, exp_id)) % 2**32`
- Same experiment ID uses same base seed across α values
- Enables paired comparisons that reduce variance
- **Note: Fixed to use zlib.crc32 in §0a**

### §3 Performance: Vectorize conglomerate stats (`ac20076`)
- Replaced per-conglomerate Python loop with `np.bincount`
- O(n_firms) instead of O(n_conglomerates × n_firms)
- Maintains bit-identical output

### Tests (`aadd0eb`)
- `test_metrics_equivalence.py`: Verifies bincount matches loop output
- `test_shock_mapping.py`: Verifies ravel() produces correct row-major order
- `test_reproducibility.py`: Verifies seeds produce identical runs
- `test_merger_rule.py`: Verifies per-member acceptance logic

### Additional Tests (Phase B §0)
- `test_seed_determinism.py`: Verifies zlib.crc32 produces same seed across processes

## Verification

Run tests to verify changes:
```bash
cd /groups/m-larch/bt307958/conglomerates_dev
module load python/3.13.3
source cong_env/bin/activate
python -m pytest tests/ -v
```

## Items Deferred to Phase B

| Item | Description | Phase B Section |
|------|-------------|-----------------|
| Analytical 1 | Growth formula derivation | §1 Growth process |
| Analytical 2 | log Δ with Δ negative | §1 Growth process |
| Calibration 4 | α=0.02 results | Rerun with fixes |
| Minor 1 | Table 2 inference | §5 Paired differences |

---

## Phase B Changes

### §1 Growth process as treatment (`TBD`)
- Added `--growth_process {normal_net, lognormal}` CLI argument
- Added `--mu_range LO HI` and `--sigma_range LO HI` CLI arguments
- Updated `model()` function signature to accept new parameters
- Implemented lognormal growth: `log δ = (μ - σ²/2) + σ·ε`
  - For lognormal: E[δ] = exp(μ), E[log δ] = μ - σ²/2
  - For normal_net (Phase A): r ~ N(μ, σ²), then log(1+r)
- Added growth_process, mu_range, sigma_range to hyperparameters output
- Created `tests/test_growth_process.py` with four tests:
  - `test_lognormal_mean_consistent`: Verifies lognormal statistical properties
  - `test_normal_net_backward_compatible`: Ensures Phase A results unchanged
  - `test_growth_processes_differ`: Confirms different processes give different results
  - `test_mu_sigma_range_respected`: Verifies parameter ranges affect output
- **Addresses Analytical 1 and Analytical 2**

### §2 Sharing rule as treatment (`TBD`)
- Added `--pooling_rule {ewp, cap}` CLI argument
  - ewp (equal-weight pooling): equal contributions and equal dollar distributions (Phase A default)
  - cap (capitalization-weighted): size-weighted contributions and return distributions
- Added `--pool_history {full, rolling}` CLI argument
- Added `--pool_window INT` CLI argument (defaults to lookback)
- Updated `process_conglomerate_pooling_numba()` to accept pooling_rule_code
  - Code 0 = ewp: avg_gain = mean(r), delta += pool / (K * w)
  - Code 1 = cap: avg_gain = sum(w * r), delta += pool
- Added pooling_rule, pool_history, pool_window to hyperparameters output
- Created `tests/test_sharing_rule.py` with six tests:
  - `test_ewp_is_default`: Verifies default is ewp
  - `test_ewp_vs_cap_differ`: Confirms different rules give different results
  - `test_ewp_backward_compatible`: Ensures Phase A results unchanged
  - `test_pool_window_recorded`: Verifies pool_window is captured
  - `test_pool_window_defaults_to_lookback`: Verifies default behavior
  - `test_pool_history_recorded`: Verifies pool_history is captured

### §3 Correlation structure as treatment (`TBD`)
- Added `--rho {uncorr, pos, neg}` CLI argument for within-market correlation
  - uncorr: ρ=0 (independent firms, Phase A default)
  - pos: ρ=0.3 (positive correlation)
  - neg: ρ=-0.3 (negative correlation)
- Added `--cross_corr {block, ar1, none}` CLI argument for cross-market correlation
  - none: identity matrix (Phase A default)
  - block: industry groups with 0.5 within-group correlation
  - ar1: distance decay with coefficient 0.7
- Implemented within-market correlation using factor model:
  - z_i = sign(ρ) * sqrt(|ρ|) * z_common + sqrt(1-|ρ|) * z_idio
- Added rho, cross_corr to hyperparameters output
- Created `tests/test_correlation.py` with six tests:
  - `test_uncorr_is_default`: Verifies default is uncorr
  - `test_none_cross_corr_is_default`: Verifies default is none
  - `test_rho_values_differ`: Confirms different rho values give different results
  - `test_cross_corr_values_differ`: Confirms different cross_corr values give different results
  - `test_uncorr_backward_compatible`: Ensures Phase A results unchanged
  - `test_positive_rho_increases_variance`: Documents variance effect

### §4 Campaign redesign (`TBD`)
- Updated α grid from np.arange(0, 0.52, 0.02) to [0.00, 0.02, 0.05] + [0.10...0.50 by 0.05]
  - 12 values total instead of 26 (fewer redundant interior points)
  - Retains key values: 0 (baseline), 0.02 (diagnostic), 0.05 (fine resolution near zero)
- Added `--mobility_csv` CLI argument for online mobility tracking
- Implemented rank autocorrelation computation in main loop:
  - Spearman rank correlation: ρ = 1 - (6 * Σd²) / (n * (n² - 1))
  - Written to CSV at each step (step > 0)
  - Measures rank persistence between consecutive timesteps
- Created `tests/test_online_mobility.py` with four tests:
  - `test_mobility_csv_created`: Verifies CSV file is created and non-empty
  - `test_rank_autocorr_valid_range`: Confirms values in [-1, 1]
  - `test_rank_autocorr_positive`: Documents typical positive autocorrelation
  - `test_no_csv_without_arg`: Verifies no side effects when not requested

### §5 Inference: Paired differences (`TBD`)
- Created `paired_analysis.py` for proper statistical inference
  - Uses common random numbers (CRN) design: same experiment_id = same base seed across α
  - Matches experiments by experiment_id for paired comparisons
  - Computes treatment effect as difference: (treatment α>0) - (control α=0)
  - Paired t-test eliminates between-replication variance
- Added `'individual_results'` to aggregated results for experiment-level access
- Added `'gini_avg'` to aggregated results (time series average across experiments)
- Created `tests/test_paired_analysis.py` with five tests:
  - `test_match_experiments_perfect`: Verifies full matching when IDs align
  - `test_match_experiments_partial`: Verifies partial matching with missing IDs
  - `test_compute_paired_differences`: Verifies difference computation
  - `test_paired_t_test_known_values`: Verifies t-test with known results
  - `test_paired_t_test_significance`: Verifies significance detection
- **Addresses Minor 1: Table 2 inference (OLS on time series)**

### §6 Burn-in check (`TBD`)
- Created `burnin_check.py` for stationarity diagnostics
  - Geweke test: compares early vs late portion means
  - Effective sample size (ESS): accounts for autocorrelation
  - Rolling mean/variance stability checks
- Provides burn-in recommendation based on convergence criteria
- Created `tests/test_burnin_check.py` with six tests:
  - `test_geweke_stationary`: Verifies convergence for stationary series
  - `test_geweke_nonstationary`: Verifies detection of trending series
  - `test_effective_sample_size_iid`: ESS equals n for IID
  - `test_effective_sample_size_autocorr`: ESS < n for AR(1)
  - `test_analyze_burnin_stationary`: Full analysis for stationary series
  - `test_recommend_burnin`: Minimal burn-in for already stationary data

### §7 Pilot design (`TBD`)
- Created `pilot_design.py` for factorial experiment generation
  - Factors: growth_process (2) × pooling_rule (2) × rho (3) × cross_corr (3) = 36 scenarios
  - α values: 12 (updated grid from §4)
  - Replications: 5 per cell (configurable)
  - Total: 36 × 12 × 5 = 2,160 experiments
- Features:
  - `--output pilot_scenarios.csv`: Generate scenario table
  - `--full_table`: Expand to full experiment table
  - `--slurm`: Generate SLURM job scripts
  - `--estimate`: Runtime estimates
- Created `tests/test_pilot_design.py` with six tests:
  - `test_factorial_design_count`: Verifies scenario count
  - `test_factorial_design_completeness`: All combinations present
  - `test_experiment_table_expansion`: Correct dimensions
  - `test_experiment_ids_unique`: Unique experiment IDs
  - `test_alpha_values_correct`: Matches updated grid
  - `test_runtime_estimate`: Positive estimates

### §8 Deliverables (`TBD`)
- MODEL_FLAGS.md: Documents all CLI arguments and their defaults
- Test suite: 8 test files covering all Phase B changes
- CHANGES_REFEREE.md: This changelog
