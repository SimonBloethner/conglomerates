# Model Flags Reference

This document describes all CLI arguments for `parallel_counterfactuals.py`.

## Core Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--markets` | int | 20 | Number of markets |
| `--firms_per_market` | int | 10 | Firms per market |
| `--steps` | int | 200 | Simulation timesteps |
| `--counterfactuals` | int | 50 | Replications per α value |
| `--share` | float | None | Single α to run (if specified) |
| `--n_cores` | int | 64 | CPU cores for parallelization |
| `--chunk` | str | None | Experiment chunk (format: start_end) |
| `--output_dir` | str | 'results' | Output directory |
| `--scenario_name` | str | None | Scenario identifier |

## Merger Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--merge_thresh` | float | 0.05 | Merger proposal threshold |
| `--comparison` | int | 4 | Number of firms to compare |
| `--break_thresh` | float | 0.85 | Conglomerate breakup threshold |
| `--proportional` | bool | False | Use proportional costs |
| `--lookback` | int | 50 | Lookback window for decisions |

## Cost Function Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cost_type` | str | 'power_law' | Cost function type |
| `--c0` | float | None | Cost level parameter |
| `--c1` | float | None | Cost shape parameter |
| `--c2` | float | None | Cost curvature parameter |

## Phase B Treatment Parameters

### §1 Growth Process

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--growth_process` | str | 'normal_net' | Growth model: 'normal_net' (Phase A) or 'lognormal' |
| `--mu_range` | float×2 | [0.01, 0.1] | Mean growth rate bounds (min, max) |
| `--sigma_range` | float×2 | [0.01, 0.05] | Volatility bounds (min, max) |

**Details:**
- `normal_net`: r ~ N(μ, σ²), store log(1+r). Phase A default.
- `lognormal`: log δ = (μ - σ²/2) + σ·ε. Proper log-growth with E[δ] = exp(μ).

### §2 Sharing Rule

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pooling_rule` | str | 'ewp' | Pooling rule: 'ewp' (equal-weight) or 'cap' (cap-weighted) |
| `--pool_history` | str | 'rolling' | Pool history: 'rolling' or 'full' |
| `--pool_window` | int | None | Rolling window size (default: same as lookback) |

**Details:**
- `ewp` (equal-weight pooling): Equal contributions, equal dollar distributions. Phase A default.
- `cap` (capitalization-weighted): Size-weighted contributions and return distributions.

### §3 Correlation Structure

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rho` | str | 'uncorr' | Within-market correlation: 'uncorr', 'pos', 'neg' |
| `--cross_corr` | str | 'none' | Cross-market correlation: 'none', 'block', 'ar1' |
| `--market_corr` | str | 'identity' | Market correlation matrix: 'identity' or 'random' |

**Details:**
- `uncorr`: ρ=0, independent firms within market. Phase A default.
- `pos`: ρ=0.3, positive correlation.
- `neg`: ρ=-0.3, negative correlation.
- `block`: Industry groups with 0.5 within-group correlation.
- `ar1`: Distance decay with coefficient 0.7.

### §4 Online Mobility

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mobility_csv` | str | None | Path to write online mobility metrics (rank autocorrelation) |

**Details:**
- If provided, writes CSV with columns: step, rank_autocorr
- Spearman rank correlation between consecutive timesteps
- Measures rank persistence (high = low mobility)

## α Grid (Phase B)

The updated α grid contains 12 values:
```
[0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
```

- 0.00: Baseline (no sharing)
- 0.02: Diagnostic (near-zero sharing)
- 0.05: Fine resolution near zero
- 0.10-0.50: Coarse grid by 0.05

## Examples

### Phase A Backward-Compatible Run
```bash
python parallel_counterfactuals.py \
    --growth_process normal_net \
    --pooling_rule ewp \
    --rho uncorr \
    --cross_corr none
```

### Phase B Lognormal with Cap-Weighted Pooling
```bash
python parallel_counterfactuals.py \
    --growth_process lognormal \
    --pooling_rule cap \
    --rho pos \
    --cross_corr block \
    --mobility_csv mobility.csv
```

### Single α with SLURM
```bash
python parallel_counterfactuals.py \
    --share 0.10 \
    --scenario_name lognormal_cap_pos_block \
    --n_cores 64 \
    --output_dir results/pilot
```

## Outputs

Results are saved as pickle files containing:
- `share`: α value
- `n_experiments`: Number of replications
- `hyperparameters`: All parameter values
- `individual_results`: List of per-experiment results (for paired analysis)
- `gini_avg`: Average Gini coefficient time series
- `mean_members_avg`: Average conglomerate size time series
- Various other aggregated metrics

## Analysis Scripts

| Script | Description |
|--------|-------------|
| `paired_analysis.py` | Paired difference t-tests using CRN |
| `burnin_check.py` | Stationarity diagnostics (Geweke test, ESS) |
| `pilot_design.py` | Factorial design generator for Phase B |
