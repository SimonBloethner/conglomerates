#!/usr/bin/env python3
"""
C10: Burn-in rerun with g=0.02.

Runs burn-in scenarios and computes convergence times for:
- hill_exponent (pooled, within-market)
- mean K
- K_eff
- cong_capital_share
- floor_hit_rate

Updates scenarios.json with new T, burn_in, and g=0.02.
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

    Parameters:
    - series: 1D array of metric values (one per metric_every steps)
    - window: rolling window size in steps (default 500)
    - tolerance: relative tolerance (default 0.02 = 2%)
    - final_window: steps for final mean computation (default 2000)
    - metric_every: steps between metric observations (default 100)

    Returns:
    - step: first step at which criterion is met (in simulation steps, not indices)
            Returns len(series)*metric_every if never converged
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


def run_burnin_scenario(alpha, seed, M=50, N=50, T=8000, metric_every=100, g=0.02):
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
        False,      # proportional
        50,         # lookback
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
        floor_c=0.05660377358490576,
        cross_corr=0.0,
        metric_every=metric_every,
        sharing_rule='proportional',
        burn_in=0,
        alpha_endogenous=False,
        g=g,
        renorm_every=500,
    )

    return result


def extract_floor_hit_rate(hyperparams, metric_every):
    """Extract floor-hit rate as a time series at metric_every intervals."""
    floor_hits_by_status = hyperparams['floor_hits_by_status']
    steps = floor_hits_by_status.shape[0]
    total_firms = hyperparams['markets'] * hyperparams['firms_per_market']

    n_obs = steps // metric_every
    floor_hit_rate = np.zeros(n_obs)

    for i in range(n_obs):
        start = i * metric_every
        end = (i + 1) * metric_every
        total_hits = floor_hits_by_status[start:end, :].sum()
        firm_periods = metric_every * total_firms
        floor_hit_rate[i] = total_hits / firm_periods if firm_periods > 0 else 0.0

    return floor_hit_rate


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


def main():
    print("C10: Burn-in rerun with g=0.02")
    print("=" * 60)

    # Parameters
    M, N, T = 50, 50, 8000
    metric_every = 100
    n_reps = 5
    base_seed = 42
    g_val = 0.02
    floor_c = 0.05660377358490576

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
            'fhr': np.zeros((n_reps, n_obs)),
        }

        print(f"\nRunning alpha={alpha} scenarios ({n_reps} reps)...")
        for rep in range(n_reps):
            seed = base_seed + rep
            print(f"  Rep {rep}: seed={seed}", end=" ", flush=True)
            result = run_burnin_scenario(alpha=alpha, seed=seed, M=M, N=N, T=T,
                                         metric_every=metric_every, g=g_val)
            hp = result[12]

            results[alpha_key]['hill'][rep] = hp['hill_exponent']
            results[alpha_key]['ccs'][rep] = hp['cong_capital_share']
            results[alpha_key]['fhr'][rep] = extract_floor_hit_rate(hp, metric_every)
            mean_k, k_eff = extract_mean_k_and_keff(hp, metric_every)
            results[alpha_key]['mean_k'][rep] = mean_k
            results[alpha_key]['k_eff'][rep] = k_eff
            print("done")

    # Compute means across reps and convergence steps
    convergence_steps = {}
    final_means = {}

    for alpha_key in ['alpha_0.0', 'alpha_0.1']:
        convergence_steps[alpha_key] = {}
        final_means[alpha_key] = {}

        for metric in ['hill', 'mean_k', 'k_eff', 'ccs', 'fhr']:
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

    print(f"\n{'='*60}")
    print("Convergence Steps (500-step rolling mean, 2% tolerance)")
    print(f"{'='*60}")

    for alpha_key in ['alpha_0.0', 'alpha_0.1']:
        print(f"\n{alpha_key}:")
        for metric in ['hill', 'mean_k', 'k_eff', 'ccs', 'fhr']:
            print(f"  {metric}: {convergence_steps[alpha_key][metric]}")

    print(f"\nburn_in = {burn_in} (max of all)")
    print(f"T = {T_new} (burn_in + 3000)")

    # Report Hill exponent against theoretical
    barrier_prediction = 1.0 / (1.0 - floor_c)  # ~1.060
    hill_alpha0_final = final_means['alpha_0.0']['hill']
    gap_pct = 100 * abs(hill_alpha0_final - barrier_prediction) / barrier_prediction

    print(f"\nalpha=0 Hill exponent (final 2000 steps):")
    print(f"  Measured: {hill_alpha0_final:.4f}")
    print(f"  Theoretical 1/(1-c): {barrier_prediction:.4f}")
    print(f"  Gap: {gap_pct:.2f}%")

    # Check acceptance assertions
    print(f"\n{'='*60}")
    print("Acceptance Assertions")
    print(f"{'='*60}")

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

    # Hill exponent bounds
    if 0.7 < hill_alpha0_final < 1.5:
        print(f"OK: alpha=0 hill_exponent = {hill_alpha0_final:.4f} in (0.7, 1.5)")
    else:
        print(f"FAIL: alpha=0 hill_exponent = {hill_alpha0_final:.4f} not in (0.7, 1.5)")

    hill_alpha01_final = final_means['alpha_0.1']['hill']
    mean_k_alpha01_final = final_means['alpha_0.1']['mean_k']

    if 0.5 < hill_alpha01_final < 1.5:
        print(f"OK: alpha=0.1 hill_exponent = {hill_alpha01_final:.4f} in (0.5, 1.5)")
    else:
        print(f"FAIL: alpha=0.1 hill_exponent = {hill_alpha01_final:.4f} not in (0.5, 1.5)")

    if mean_k_alpha01_final > 1.0:
        print(f"OK: alpha=0.1 mean_k = {mean_k_alpha01_final:.4f} > 1.0 (conglomerates form)")
    else:
        print(f"FAIL: alpha=0.1 mean_k = {mean_k_alpha01_final:.4f} <= 1.0 (no conglomerates)")

    # Write CSV
    csv_path = 'diagnostics/burn_in_c.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'alpha', 'hill', 'mean_k', 'k_eff', 'ccs', 'fhr'])
        steps_axis = np.arange(1, n_obs + 1) * metric_every
        for alpha_key, alpha_val in [('alpha_0.0', 0.0), ('alpha_0.1', 0.1)]:
            for i, step in enumerate(steps_axis):
                row = [
                    step, alpha_val,
                    np.nanmean(results[alpha_key]['hill'][:, i]),
                    np.nanmean(results[alpha_key]['mean_k'][:, i]),
                    np.nanmean(results[alpha_key]['k_eff'][:, i]),
                    np.nanmean(results[alpha_key]['ccs'][:, i]),
                    np.nanmean(results[alpha_key]['fhr'][:, i]),
                ]
                writer.writerow(row)
    print(f"\nWrote {csv_path}")

    # Create figures (one per alpha)
    steps_axis = np.arange(1, n_obs + 1) * metric_every

    for alpha_key, alpha_val, filename in [('alpha_0.0', 0.0, 'burnin_c_alpha0.png'),
                                            ('alpha_0.1', 0.1, 'burnin_c_alpha01.png')]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Burn-in Metrics: alpha={alpha_val}, g={g_val}', fontsize=14)

        metrics_info = [
            ('hill', 'Hill exponent (pooled)', axes[0, 0]),
            ('mean_k', 'Mean K', axes[0, 1]),
            ('k_eff', 'K_eff', axes[0, 2]),
            ('ccs', 'Cong capital share', axes[1, 0]),
            ('fhr', 'Floor-hit rate', axes[1, 1]),
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
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty subplot
        axes[1, 2].axis('off')

        plt.tight_layout()
        fig_path = f'diagnostics/figures/{filename}'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Wrote {fig_path}")

    # Write markdown report
    report = f"""# C10: Burn-in Rerun with g=0.02

## Setup

**Parameters**:
- M = N = {M}
- g = {g_val} (common time-average growth)
- sigma_range = (0.1, 0.3)
- floor_c = {floor_c:.4f} (exponent 1/(1-c) = {barrier_prediction:.3f})
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
| Floor-hit rate | {convergence_steps['alpha_0.0']['fhr']} | {final_means['alpha_0.0']['fhr']:.4f} |

### alpha=0.1

| Metric | Convergence Step | Final Mean |
|--------|-----------------|------------|
| Hill exponent | {convergence_steps['alpha_0.1']['hill']} | {final_means['alpha_0.1']['hill']:.4f} |
| Mean K | {convergence_steps['alpha_0.1']['mean_k']} | {final_means['alpha_0.1']['mean_k']:.2f} |
| K_eff | {convergence_steps['alpha_0.1']['k_eff']} | {final_means['alpha_0.1']['k_eff']:.2f} |
| Cong capital share | {convergence_steps['alpha_0.1']['ccs']} | {final_means['alpha_0.1']['ccs']:.4f} |
| Floor-hit rate | {convergence_steps['alpha_0.1']['fhr']} | {final_means['alpha_0.1']['fhr']:.4f} |

## Burn-in Recommendation

**burn_in = {burn_in}** (max of all convergence steps)

**T = {T_new}** (burn_in + 3000)

## Hill Exponent vs Theoretical

**Theoretical barrier**: 1/(1-c) = 1/(1-{floor_c:.4f}) = {barrier_prediction:.4f}

**Measured** (alpha=0, final 2000 steps): {hill_alpha0_final:.4f}

**Gap**: {gap_pct:.2f}%

## Metric Tables (every 1000 steps)

### alpha=0.0

| Step | Hill | Mean K | K_eff | Cong Capital Share | Floor-hit Rate |
|------|------|--------|-------|-------------------|----------------|
"""

    # Add table rows for alpha=0
    for step in range(1000, T + 1, 1000):
        idx = step // metric_every - 1
        if idx < n_obs and idx >= 0:
            report += f"| {step} | {np.nanmean(results['alpha_0.0']['hill'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.0']['mean_k'][:, idx]):.2f} | "
            report += f"{np.nanmean(results['alpha_0.0']['k_eff'][:, idx]):.2f} | "
            report += f"{np.nanmean(results['alpha_0.0']['ccs'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.0']['fhr'][:, idx]):.4f} |\n"

    report += """
### alpha=0.1

| Step | Hill | Mean K | K_eff | Cong Capital Share | Floor-hit Rate |
|------|------|--------|-------|-------------------|----------------|
"""

    # Add table rows for alpha=0.1
    for step in range(1000, T + 1, 1000):
        idx = step // metric_every - 1
        if idx < n_obs and idx >= 0:
            report += f"| {step} | {np.nanmean(results['alpha_0.1']['hill'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.1']['mean_k'][:, idx]):.2f} | "
            report += f"{np.nanmean(results['alpha_0.1']['k_eff'][:, idx]):.2f} | "
            report += f"{np.nanmean(results['alpha_0.1']['ccs'][:, idx]):.4f} | "
            report += f"{np.nanmean(results['alpha_0.1']['fhr'][:, idx]):.4f} |\n"

    report += f"""
## Figures

- `figures/burnin_c_alpha0.png`: All metrics for alpha=0.0
- `figures/burnin_c_alpha01.png`: All metrics for alpha=0.1
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
        scenario['g'] = g_val
        scenario['mu_range'] = None  # Set to null as per card

    with open('pilot_c/scenarios.json', 'w') as f:
        json.dump(scenarios, f, indent=2)
    print(f"Updated {len(scenarios)} scenarios with T={T_new}, burn_in={burn_in}, g={g_val}")

    print("\nDone!")
    return burn_in, T_new, hill_alpha0_final, gap_pct


if __name__ == '__main__':
    main()
