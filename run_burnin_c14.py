#!/usr/bin/env python3
"""
C18: Burn-in with decision_rule=loggain.

Runs burn-in scenarios for:
- Main block, laplace, power_law
- M=N=50, IQR ∈ [0.1, 0.3], proportional sharing
- T=8000, 5 reps
- alpha=0 and alpha=0.1
- market_size_fixed=True, floor_c=c_star=0.12717
- decision_rule='loggain' (C18 update)

Series tracked:
- hill_exponent, mean K, K_eff, cong_capital_share
- floor-hit rate (members, standalones)
- member-standalone growth gap
- mergers_per_period (C18)

Convergence step per series; updates scenarios.json with burn_in and T.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import csv

sys.path.insert(0, '.')
from collaborative_growth import model, seed_numba


def compute_convergence_step(series, window=500, tolerance=0.02, final_window=2000, metric_every=100):
    """
    Find the first step at which the rolling mean stays within tolerance of the final mean.
    """
    # Convert windows from steps to indices
    window_idx = window // metric_every
    final_window_idx = final_window // metric_every

    # Find valid range (non-NaN values)
    valid_mask = ~np.isnan(series)
    if not valid_mask.any():
        return len(series) * metric_every

    # Truncate to valid range for analysis
    last_valid_idx = np.where(valid_mask)[0][-1]
    series_valid = series[:last_valid_idx + 1]

    if len(series_valid) < final_window_idx:
        final_window_idx = len(series_valid) // 2

    if final_window_idx < window_idx:
        return len(series) * metric_every

    # Compute final mean from last final_window_idx observations of valid data
    final_mean = np.nanmean(series_valid[-final_window_idx:])

    if np.isnan(final_mean):
        return len(series) * metric_every

    # Handle constant-zero series
    if final_mean == 0 and np.all(series_valid == 0):
        return window

    # For near-zero final mean, use absolute tolerance
    use_absolute = abs(final_mean) < 1e-10

    # Scan forward to find first point where rolling mean stays within tolerance
    n_valid = len(series_valid)
    for i in range(window_idx - 1, n_valid):
        start_idx = max(0, i - window_idx + 1)
        rolling_mean = np.nanmean(series_valid[start_idx:i + 1])

        if np.isnan(rolling_mean):
            continue

        if use_absolute:
            rel_error = abs(rolling_mean - final_mean)
        else:
            rel_error = abs(rolling_mean - final_mean) / abs(final_mean)

        if rel_error <= tolerance:
            # Check that it stays within tolerance for rest of valid data
            stays_converged = True
            for j in range(i, n_valid):
                start_j = max(0, j - window_idx + 1)
                rm_j = np.nanmean(series_valid[start_j:j + 1])
                if np.isnan(rm_j):
                    continue
                if use_absolute:
                    err_j = abs(rm_j - final_mean)
                else:
                    err_j = abs(rm_j - final_mean) / abs(final_mean)
                if err_j > tolerance:
                    stays_converged = False
                    break

            if stays_converged:
                return (i + 1) * metric_every

    return n_valid * metric_every


def run_burnin_scenario(alpha, seed, M=50, N=50, T=8000, metric_every=100, g=0.02,
                        floor_c=0.12717, lookback=100, decision_rule='loggain'):
    """Run a single burn-in scenario."""
    total_firms = M * N

    params = [
        M,          # markets
        N,          # firms_per_market
        T,          # steps
        alpha,      # share (alpha)
        total_firms,
        0.05,       # merge_thresh
        4,          # K_max (comparison)
        0.0,        # minimum_benefit
        False,      # proportional (param - will use sharing_rule='proportional')
        lookback,   # lookback (set to 100 for growth gap computation)
        'power_law',
        None, None, None,  # c0, c1, c2 defaults
    ]

    result = model(
        params,
        seed=seed,
        growth_process='log_family',
        log_family='laplace',
        mu_range=(0.01, 0.1),  # Ignored when g is set
        sigma_range=(0.1, 0.3),
        floor_c=floor_c,
        cross_corr=0.0,
        metric_every=metric_every,
        sharing_rule='proportional',
        burn_in=0,
        alpha_endogenous=False,
        g=g,
        renorm_every=500,
        market_size_fixed=True,  # C14: fixed market size
        decision_rule=decision_rule,  # C18: loggain by default
    )

    return result


def extract_floor_hit_rate_by_status(hyperparams, metric_every):
    """Extract floor-hit rate by status (standalone=0, member=1) as time series."""
    floor_hits_by_status = hyperparams['floor_hits_by_status']
    steps = floor_hits_by_status.shape[0]
    total_firms = hyperparams['markets'] * hyperparams['firms_per_market']

    n_obs = steps // metric_every
    fhr_standalone = np.zeros(n_obs)
    fhr_member = np.zeros(n_obs)

    for i in range(n_obs):
        start = i * metric_every
        end = (i + 1) * metric_every
        standalone_hits = floor_hits_by_status[start:end, 0].sum()
        member_hits = floor_hits_by_status[start:end, 1].sum()
        firm_periods = metric_every * total_firms
        fhr_standalone[i] = standalone_hits / firm_periods if firm_periods > 0 else 0.0
        fhr_member[i] = member_hits / firm_periods if firm_periods > 0 else 0.0

    return fhr_standalone, fhr_member


def extract_mean_k_and_keff(hyperparams, metric_every):
    """Extract mean K and K_eff as time series at metric_every intervals."""
    effective_members_list = hyperparams['effective_members']
    steps = hyperparams['steps']
    n_obs = steps // metric_every

    mean_k = np.zeros(n_obs)
    mean_k_eff = np.zeros(n_obs)

    # Build a mapping from step to K and K_eff values
    step_to_data = {}
    for step_val, step_eff in effective_members_list:
        ks = [K for (cid, K, K_eff) in step_eff]
        k_effs = [K_eff for (cid, K, K_eff) in step_eff]
        step_to_data[step_val] = (ks, k_effs)

    for i in range(n_obs):
        step = (i + 1) * metric_every - 1  # Last step in this interval
        if step in step_to_data:
            ks, k_effs = step_to_data[step]
            mean_k[i] = np.mean(ks) if len(ks) > 0 else 0.0
            mean_k_eff[i] = np.mean(k_effs) if len(k_effs) > 0 else 0.0
        else:
            mean_k[i] = 0.0
            mean_k_eff[i] = 0.0

    return mean_k, mean_k_eff


def compute_growth_gap(hyperparams, metric_every):
    """
    Compute member-standalone growth gap at each metric interval.

    For each market, compute mean log share change over the last 100 steps
    for conglomerate members vs standalones, then average across markets.

    Returns:
    - growth_gap: array of shape (n_obs,) with gap at each metric step
                  Positive = members grew faster than standalones
    """
    # Note: With market_size_fixed=True, log_state changes represent log share changes
    # The model tracks floor_hits_by_status but doesn't track log states at each metric step
    # For this analysis, we use a simplified approach based on final states

    # Since the model doesn't return log states at each metric step,
    # we compute a proxy: the difference in floor-hit rates scaled by typical floor impact
    # Firms hitting the floor have negative growth; those not hitting have positive/neutral growth

    # More accurate approach would require model modification, but this proxy
    # captures the key signal: members vs standalones hitting the floor at different rates

    floor_hits_by_status = hyperparams['floor_hits_by_status']
    steps = floor_hits_by_status.shape[0]
    n_obs = steps // metric_every

    growth_gap = np.zeros(n_obs)

    # Get firm membership at final state (approximation: membership is stable at equilibrium)
    firm_conglom = hyperparams['final_firm_conglom']
    total_firms = hyperparams['markets'] * hyperparams['firms_per_market']

    # Count members and standalones
    n_members = np.sum(firm_conglom >= 0)
    n_standalones = total_firms - n_members

    for i in range(n_obs):
        start = i * metric_every
        end = (i + 1) * metric_every

        standalone_hits = floor_hits_by_status[start:end, 0].sum()
        member_hits = floor_hits_by_status[start:end, 1].sum()

        # Floor-hit rate per firm per step
        fhr_standalone = standalone_hits / (n_standalones * metric_every) if n_standalones > 0 else 0.0
        fhr_member = member_hits / (n_members * metric_every) if n_members > 0 else 0.0

        # Growth gap proxy: negative of floor-hit rate difference
        # Lower floor-hit rate -> higher (less negative) growth
        # Gap = member_growth - standalone_growth
        #     ~ -fhr_member - (-fhr_standalone)
        #     = fhr_standalone - fhr_member
        growth_gap[i] = fhr_standalone - fhr_member

    return growth_gap


def main():
    print("C18: Burn-in with decision_rule=loggain")
    print("=" * 70)

    # Parameters
    M, N, T = 50, 50, 8000
    metric_every = 100
    n_reps = 5
    base_seed = 42
    g_val = 0.02
    floor_c = 0.12717  # c_star from C13
    lookback = 100  # For growth gap computation

    # Create output directories
    Path('diagnostics/figures').mkdir(parents=True, exist_ok=True)

    n_obs = T // metric_every

    # Storage for results
    results = {}
    for alpha in [0.0, 0.1]:
        alpha_key = f"alpha_{alpha}"
        results[alpha_key] = {
            'hill': np.zeros((n_reps, n_obs)),
            'mean_k': np.zeros((n_reps, n_obs)),
            'k_eff': np.zeros((n_reps, n_obs)),
            'ccs': np.zeros((n_reps, n_obs)),
            'fhr_standalone': np.zeros((n_reps, n_obs)),
            'fhr_member': np.zeros((n_reps, n_obs)),
            'growth_gap': np.zeros((n_reps, n_obs)),
            'mergers_per_period': np.zeros(n_reps),  # C18: track mergers
        }

        print(f"\nRunning alpha={alpha} scenarios ({n_reps} reps)...")
        for rep in range(n_reps):
            seed = base_seed + rep
            print(f"  Rep {rep}: seed={seed}", end=" ", flush=True)
            result = run_burnin_scenario(alpha=alpha, seed=seed, M=M, N=N, T=T,
                                         metric_every=metric_every, g=g_val,
                                         floor_c=floor_c, lookback=lookback)
            hp = result[-1]  # hyperparameters dict

            results[alpha_key]['hill'][rep] = hp['hill_exponent']
            results[alpha_key]['ccs'][rep] = hp['cong_capital_share']

            fhr_standalone, fhr_member = extract_floor_hit_rate_by_status(hp, metric_every)
            results[alpha_key]['fhr_standalone'][rep] = fhr_standalone
            results[alpha_key]['fhr_member'][rep] = fhr_member

            mean_k, k_eff = extract_mean_k_and_keff(hp, metric_every)
            results[alpha_key]['mean_k'][rep] = mean_k
            results[alpha_key]['k_eff'][rep] = k_eff

            growth_gap = compute_growth_gap(hp, metric_every)
            results[alpha_key]['growth_gap'][rep] = growth_gap

            # C18: Track mergers_per_period from summary
            summary = hp.get('summary', {})
            results[alpha_key]['mergers_per_period'][rep] = summary.get('mergers_per_period', np.nan)

            print("done")

    # Compute means across reps and convergence steps
    convergence_steps = {}
    final_means = {}

    for alpha_key in ['alpha_0.0', 'alpha_0.1']:
        convergence_steps[alpha_key] = {}
        final_means[alpha_key] = {}

        for metric in ['hill', 'mean_k', 'k_eff', 'ccs', 'fhr_standalone', 'fhr_member', 'growth_gap']:
            mean_series = np.nanmean(results[alpha_key][metric], axis=0)
            conv_step = compute_convergence_step(mean_series, window=500, tolerance=0.02,
                                                 final_window=2000, metric_every=metric_every)
            convergence_steps[alpha_key][metric] = conv_step

            # Final 2000-step mean
            final_idx = 2000 // metric_every
            final_means[alpha_key][metric] = np.nanmean(mean_series[-final_idx:])

    # Find global burn_in
    all_conv = []
    for alpha_key in convergence_steps:
        for metric in convergence_steps[alpha_key]:
            all_conv.append(convergence_steps[alpha_key][metric])

    burn_in = max(all_conv)
    T_new = burn_in + 3000

    print(f"\n{'='*70}")
    print("Convergence Steps (500-step rolling mean, 2% tolerance)")
    print(f"{'='*70}")

    for alpha_key in ['alpha_0.0', 'alpha_0.1']:
        print(f"\n{alpha_key}:")
        for metric in ['hill', 'mean_k', 'k_eff', 'ccs', 'fhr_standalone', 'fhr_member', 'growth_gap']:
            print(f"  {metric}: {convergence_steps[alpha_key][metric]}")

    print(f"\nburn_in = {burn_in} (max of all)")
    print(f"T = {T_new} (burn_in + 3000)")

    # Report Hill exponent against theoretical
    theoretical_hill = 1.06
    hill_alpha0_final = final_means['alpha_0.0']['hill']
    gap_pct = 100 * abs(hill_alpha0_final - theoretical_hill) / theoretical_hill

    print(f"\nalpha=0 Hill exponent (final 2000 steps):")
    print(f"  Measured: {hill_alpha0_final:.4f}")
    print(f"  Target: {theoretical_hill:.4f}")
    print(f"  Gap: {gap_pct:.2f}%")

    # Check acceptance assertions
    print(f"\n{'='*70}")
    print("Acceptance Assertions")
    print(f"{'='*70}")

    # Check for NaN/inf
    has_nan_inf = False
    for alpha_key in results:
        for metric in results[alpha_key]:
            data = results[alpha_key][metric]
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(f"WARNING: NaN/inf in {alpha_key} {metric}")
                has_nan_inf = True

    if has_nan_inf:
        print("FAIL: NaN or inf detected in series")
    else:
        print("OK: No NaN or inf in any series")

    # alpha=0 Hill within 15% of 1.06
    if abs(hill_alpha0_final - theoretical_hill) / theoretical_hill <= 0.15:
        print(f"OK: alpha=0 hill_exponent = {hill_alpha0_final:.4f} within 15% of {theoretical_hill}")
    else:
        print(f"FAIL: alpha=0 hill_exponent = {hill_alpha0_final:.4f} NOT within 15% of {theoretical_hill}")

    # alpha=0.1 assertions (C18 thresholds)
    ccs_alpha01 = final_means['alpha_0.1']['ccs']
    mean_k_alpha01 = final_means['alpha_0.1']['mean_k']
    k_eff_alpha01 = final_means['alpha_0.1']['k_eff']
    k_ratio = k_eff_alpha01 / mean_k_alpha01 if mean_k_alpha01 > 0 else 0.0
    mergers_alpha01 = np.nanmean(results['alpha_0.1']['mergers_per_period'])

    # C18: mean K >= 3.0 (vs 2.5 for replay)
    if mean_k_alpha01 >= 3.0:
        print(f"OK: alpha=0.1 mean_k = {mean_k_alpha01:.4f} >= 3.0")
    else:
        print(f"FAIL: alpha=0.1 mean_k = {mean_k_alpha01:.4f} NOT >= 3.0 (loggain should drive larger conglomerates)")

    # C18: ccs > 0.05 (vs 0.02 for replay)
    if ccs_alpha01 > 0.05:
        print(f"OK: alpha=0.1 cong_capital_share = {ccs_alpha01:.4f} > 0.05")
    else:
        print(f"FAIL: alpha=0.1 cong_capital_share = {ccs_alpha01:.4f} NOT > 0.05")

    # C18: mergers_per_period < 2.0 (reduced churn)
    if mergers_alpha01 < 2.0:
        print(f"OK: alpha=0.1 mergers_per_period = {mergers_alpha01:.4f} < 2.0")
    else:
        print(f"FAIL: alpha=0.1 mergers_per_period = {mergers_alpha01:.4f} NOT < 2.0 (churn too high)")

    if k_ratio >= 0.5:
        print(f"OK: alpha=0.1 K_eff/K = {k_ratio:.4f} >= 0.5")
    else:
        print(f"FAIL: alpha=0.1 K_eff/K = {k_ratio:.4f} NOT >= 0.5")

    # C18: Compute mean mergers_per_period per alpha
    mergers_alpha0 = np.nanmean(results['alpha_0.0']['mergers_per_period'])
    mergers_alpha01 = np.nanmean(results['alpha_0.1']['mergers_per_period'])

    # Write CSV
    csv_path = 'diagnostics/burn_in_c.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'alpha', 'hill', 'mean_k', 'k_eff', 'ccs',
                         'fhr_standalone', 'fhr_member', 'growth_gap', 'mergers_per_period'])
        steps_axis = np.arange(1, n_obs + 1) * metric_every
        for alpha_key, alpha_val, mergers_val in [('alpha_0.0', 0.0, mergers_alpha0),
                                                   ('alpha_0.1', 0.1, mergers_alpha01)]:
            for i, step in enumerate(steps_axis):
                row = [
                    step, alpha_val,
                    np.nanmean(results[alpha_key]['hill'][:, i]),
                    np.nanmean(results[alpha_key]['mean_k'][:, i]),
                    np.nanmean(results[alpha_key]['k_eff'][:, i]),
                    np.nanmean(results[alpha_key]['ccs'][:, i]),
                    np.nanmean(results[alpha_key]['fhr_standalone'][:, i]),
                    np.nanmean(results[alpha_key]['fhr_member'][:, i]),
                    np.nanmean(results[alpha_key]['growth_gap'][:, i]),
                    mergers_val,  # C18: constant per alpha
                ]
                writer.writerow(row)
    print(f"\nWrote {csv_path}")

    # Create figures (one per alpha)
    steps_axis = np.arange(1, n_obs + 1) * metric_every

    for alpha_key, alpha_val, filename in [('alpha_0.0', 0.0, 'burnin_c_alpha0.png'),
                                            ('alpha_0.1', 0.1, 'burnin_c_alpha01.png')]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'C14 Burn-in Metrics: alpha={alpha_val}, floor_c={floor_c}, market_size_fixed=True',
                     fontsize=12)

        metrics_info = [
            ('hill', 'Hill exponent', axes[0, 0]),
            ('mean_k', 'Mean K', axes[0, 1]),
            ('k_eff', 'K_eff', axes[0, 2]),
            ('ccs', 'Cong capital share', axes[0, 3]),
            ('fhr_standalone', 'Floor-hit rate (standalone)', axes[1, 0]),
            ('fhr_member', 'Floor-hit rate (member)', axes[1, 1]),
            ('growth_gap', 'Growth gap (member - standalone)', axes[1, 2]),
        ]

        for metric, label, ax in metrics_info:
            mean_series = np.nanmean(results[alpha_key][metric], axis=0)
            conv_step = convergence_steps[alpha_key][metric]

            ax.plot(steps_axis, mean_series, 'b-', linewidth=1)
            ax.axvline(conv_step, color='r', linestyle='--', label=f'Conv @ {conv_step}')
            ax.axvline(burn_in, color='k', linestyle=':', label=f'burn_in={burn_in}')
            ax.set_xlabel('Step')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Remove empty subplot
        axes[1, 3].axis('off')

        plt.tight_layout()
        fig_path = f'diagnostics/figures/{filename}'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Wrote {fig_path}")

    # Write markdown report
    report = f"""# C18: Burn-in with decision_rule=loggain

## Setup

**Parameters**:
- M = N = {M}
- g = {g_val} (common time-average growth)
- sigma_range = (0.1, 0.3)
- floor_c = {floor_c} (c_star from C13)
- market_size_fixed = True
- decision_rule = loggain (C18)
- T = {T} for burn-in analysis
- {n_reps} replications with seeds {base_seed} to {base_seed + n_reps - 1}
- metric_every = {metric_every}
- Sharing rule: proportional
- Growth process: log_family (laplace)
- Cost function: power_law

## Convergence Results

Using 500-step rolling mean vs final 2000-step mean, 2% tolerance.

### alpha=0.0

| Metric | Convergence Step | Final Mean |
|--------|-----------------|------------|
| Hill exponent | {convergence_steps['alpha_0.0']['hill']} | {final_means['alpha_0.0']['hill']:.4f} |
| Mean K | {convergence_steps['alpha_0.0']['mean_k']} | {final_means['alpha_0.0']['mean_k']:.2f} |
| K_eff | {convergence_steps['alpha_0.0']['k_eff']} | {final_means['alpha_0.0']['k_eff']:.2f} |
| Cong capital share | {convergence_steps['alpha_0.0']['ccs']} | {final_means['alpha_0.0']['ccs']:.4f} |
| Floor-hit rate (standalone) | {convergence_steps['alpha_0.0']['fhr_standalone']} | {final_means['alpha_0.0']['fhr_standalone']:.4f} |
| Floor-hit rate (member) | {convergence_steps['alpha_0.0']['fhr_member']} | {final_means['alpha_0.0']['fhr_member']:.4f} |
| Growth gap | {convergence_steps['alpha_0.0']['growth_gap']} | {final_means['alpha_0.0']['growth_gap']:.4f} |

### alpha=0.1

| Metric | Convergence Step | Final Mean |
|--------|-----------------|------------|
| Hill exponent | {convergence_steps['alpha_0.1']['hill']} | {final_means['alpha_0.1']['hill']:.4f} |
| Mean K | {convergence_steps['alpha_0.1']['mean_k']} | {final_means['alpha_0.1']['mean_k']:.2f} |
| K_eff | {convergence_steps['alpha_0.1']['k_eff']} | {final_means['alpha_0.1']['k_eff']:.2f} |
| Cong capital share | {convergence_steps['alpha_0.1']['ccs']} | {final_means['alpha_0.1']['ccs']:.4f} |
| Floor-hit rate (standalone) | {convergence_steps['alpha_0.1']['fhr_standalone']} | {final_means['alpha_0.1']['fhr_standalone']:.4f} |
| Floor-hit rate (member) | {convergence_steps['alpha_0.1']['fhr_member']} | {final_means['alpha_0.1']['fhr_member']:.4f} |
| Growth gap | {convergence_steps['alpha_0.1']['growth_gap']} | {final_means['alpha_0.1']['growth_gap']:.4f} |

## Burn-in Recommendation

**burn_in = {burn_in}** (max of all convergence steps)

**T = {T_new}** (burn_in + 3000)

## Acceptance Assertions

### alpha=0

- Hill exponent (final 2000): {hill_alpha0_final:.4f}
- Target: 1.06
- Gap: {gap_pct:.2f}%
- Status: {"PASS" if gap_pct <= 15 else "FAIL"} (within 15%)

### alpha=0.1 (C18 loggain thresholds)

- mean K: {mean_k_alpha01:.2f} {">=" if mean_k_alpha01 >= 3.0 else "<"} 3.0 -> {"PASS" if mean_k_alpha01 >= 3.0 else "FAIL"}
- cong_capital_share: {ccs_alpha01:.4f} {">" if ccs_alpha01 > 0.05 else "<="} 0.05 -> {"PASS" if ccs_alpha01 > 0.05 else "FAIL"}
- mergers_per_period: {mergers_alpha01:.2f} {"<" if mergers_alpha01 < 2.0 else ">="} 2.0 -> {"PASS" if mergers_alpha01 < 2.0 else "FAIL"}
- K_eff/K: {k_ratio:.4f} {">=" if k_ratio >= 0.5 else "<"} 0.5 -> {"PASS" if k_ratio >= 0.5 else "FAIL"}

## Metric Tables (every 1000 steps)

### alpha=0.0

| Step | Hill | Mean K | K_eff | Cong Share | FHR Stand | FHR Memb | Gap |
|------|------|--------|-------|------------|-----------|----------|-----|
"""

    # Add table rows for alpha=0
    for step in range(1000, T + 1, 1000):
        idx = step // metric_every - 1
        if idx < n_obs and idx >= 0:
            report += f"| {step} | {np.nanmean(results['alpha_0.0']['hill'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.0']['mean_k'][:, idx]):.2f} | "
            report += f"{np.nanmean(results['alpha_0.0']['k_eff'][:, idx]):.2f} | "
            report += f"{np.nanmean(results['alpha_0.0']['ccs'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.0']['fhr_standalone'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.0']['fhr_member'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.0']['growth_gap'][:, idx]):.4f} |\n"

    report += """
### alpha=0.1

| Step | Hill | Mean K | K_eff | Cong Share | FHR Stand | FHR Memb | Gap |
|------|------|--------|-------|------------|-----------|----------|-----|
"""

    # Add table rows for alpha=0.1
    for step in range(1000, T + 1, 1000):
        idx = step // metric_every - 1
        if idx < n_obs and idx >= 0:
            report += f"| {step} | {np.nanmean(results['alpha_0.1']['hill'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.1']['mean_k'][:, idx]):.2f} | "
            report += f"{np.nanmean(results['alpha_0.1']['k_eff'][:, idx]):.2f} | "
            report += f"{np.nanmean(results['alpha_0.1']['ccs'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.1']['fhr_standalone'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.1']['fhr_member'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.1']['growth_gap'][:, idx]):.4f} |\n"

    report += f"""
## Figures

- `figures/burnin_c_alpha0.png`: All metrics for alpha=0.0
- `figures/burnin_c_alpha01.png`: All metrics for alpha=0.1

## Notes

Growth gap is computed as the difference in floor-hit rates between standalones and members.
A positive gap indicates standalones hit the floor more often (worse performance).
"""

    with open('diagnostics/burn_in_c.md', 'w') as f:
        f.write(report)
    print("Wrote diagnostics/burn_in_c.md")

    # Update scenarios.json
    print(f"\nUpdating pilot_c/scenarios.json...")
    with open('pilot_c/scenarios.json', 'r') as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        scenario['T'] = T_new
        scenario['burn_in'] = burn_in

    with open('pilot_c/scenarios.json', 'w') as f:
        json.dump(scenarios, f, indent=2)
    print(f"Updated {len(scenarios)} scenarios with T={T_new}, burn_in={burn_in}")

    print("\nDone!")
    return burn_in, T_new, final_means


if __name__ == '__main__':
    main()
