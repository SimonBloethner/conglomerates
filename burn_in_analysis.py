#!/usr/bin/env python3
"""
Phase C burn-in analysis.

Runs burn-in scenarios (main block, laplace, power_law, α=0.1) with T=8000.
Determines convergence point using 500-step rolling means within 2% of final-2000-step mean.
Outputs T and burn_in values for Phase C pilot.
"""
import json
import os
import numpy as np
from pathlib import Path

from collaborative_growth import model, seed_numba
from pilot_design import generate_all_scenarios, get_burn_in_scenario, FLOOR_C


# Burn-in parameters
T_BURNIN = 8000
WINDOW = 500
TOLERANCE = 0.02  # 2%
FINAL_WINDOW = 2000


def run_burn_in_scenario(scenario, T):
    """Run a single burn-in scenario and return metrics."""
    total_firms = scenario['M'] * scenario['N']

    params = [
        scenario['M'],
        scenario['N'],
        T,
        scenario['alpha'],
        total_firms,
        scenario['merge_thresh'],
        4,  # K_max default
        0.0,  # minimum_benefit default
        scenario['proportional'],
        scenario['lookback'],
        scenario['cost_type'],
        scenario['c0'],
        scenario['c1'],
        scenario['c2'],
    ]

    result = model(
        params,
        seed=scenario['seed'],
        growth_process=scenario['growth_process'],
        log_family=scenario['log_family'],
        mu_range=tuple(scenario['mu_range']),
        sigma_range=tuple(scenario['sigma_range']),
        floor_c=scenario['floor_c'],
        cross_corr=scenario['cross_corr'],
        metric_every=scenario['metric_every'],
        sharing_rule=scenario['sharing_rule'],
        alpha_endogenous=scenario.get('alpha_endogenous', False),
    )

    hyperparams = result[-1]

    return {
        'hill_exponent': hyperparams.get('hill_exponent'),
        'hhi_aggregate': hyperparams.get('hhi_aggregate'),
        'cong_capital_share': hyperparams.get('cong_capital_share'),
        'effective_members': hyperparams.get('effective_members'),
        'metric_every': scenario['metric_every'],
    }


def extract_mean_K(effective_members, metric_every):
    """Extract mean K time series from effective_members."""
    if not effective_members:
        return np.array([])

    # effective_members is list of (step, [(cong_id, K, K_eff), ...])
    mean_K = []
    for step, congs in effective_members:
        if congs:
            K_values = [c[1] for c in congs]
            mean_K.append(np.mean(K_values))
        else:
            mean_K.append(np.nan)

    return np.array(mean_K)


def rolling_mean(x, window):
    """Compute rolling mean with given window size."""
    if len(x) < window:
        return np.full(len(x), np.nan)

    result = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        result[i] = np.nanmean(x[i - window + 1:i + 1])

    return result


def find_convergence_step(series, metric_every, window=WINDOW, tolerance=TOLERANCE,
                          final_window=FINAL_WINDOW):
    """
    Find first step where rolling mean is within tolerance of final mean.

    Returns step number (not index).
    """
    if len(series) == 0 or np.all(np.isnan(series)):
        return None

    # Convert window and final_window from steps to indices
    window_idx = max(1, window // metric_every)
    final_idx = max(1, final_window // metric_every)

    # Final mean (last final_window steps)
    final_mean = np.nanmean(series[-final_idx:])
    if np.isnan(final_mean) or final_mean == 0:
        return None

    # Rolling mean
    roll = rolling_mean(series, window_idx)

    # Find first index where |roll - final| / |final| < tolerance
    for i in range(len(roll)):
        if not np.isnan(roll[i]):
            rel_diff = abs(roll[i] - final_mean) / abs(final_mean)
            if rel_diff < tolerance:
                # Check that it stays within tolerance for rest of series
                remaining = roll[i:]
                if np.all(np.abs(remaining[~np.isnan(remaining)] - final_mean) /
                          abs(final_mean) < tolerance):
                    return (i + 1) * metric_every

    return None


def analyze_convergence(results, metric_every):
    """Analyze convergence across all metrics and reps."""
    convergence = {
        'hill_exponent': [],
        'cong_capital_share': [],
        'mean_K': [],
    }

    for r in results:
        # Hill exponent (average across markets)
        if r['hill_exponent'] is not None:
            hill = np.nanmean(r['hill_exponent'], axis=0)  # Average over markets
            step = find_convergence_step(hill, metric_every)
            if step is not None:
                convergence['hill_exponent'].append(step)

        # Cong capital share
        if r['cong_capital_share'] is not None:
            ccs = r['cong_capital_share']
            step = find_convergence_step(ccs, metric_every)
            if step is not None:
                convergence['cong_capital_share'].append(step)

        # Mean K
        mean_K = extract_mean_K(r['effective_members'], metric_every)
        if len(mean_K) > 0:
            step = find_convergence_step(mean_K, metric_every)
            if step is not None:
                convergence['mean_K'].append(step)

    return convergence


def main():
    """Run burn-in analysis and determine T and burn_in."""
    print("Phase C Burn-in Analysis")
    print("=" * 50)

    # Generate scenarios and get burn-in subset
    scenarios = generate_all_scenarios()
    burn_in_scenarios = get_burn_in_scenario(scenarios)

    print(f"Burn-in scenarios: {len(burn_in_scenarios)}")
    print(f"  log_family: {burn_in_scenarios[0]['log_family']}")
    print(f"  cost_type: {burn_in_scenarios[0]['cost_type']}")
    print(f"  alpha: {burn_in_scenarios[0]['alpha']}")
    print(f"  floor_c: {burn_in_scenarios[0]['floor_c']:.10f}")
    print(f"  T: {T_BURNIN}")
    print()

    # Run all burn-in scenarios
    print("Running burn-in scenarios...")
    results = []
    for i, s in enumerate(burn_in_scenarios):
        print(f"  Rep {i+1}/{len(burn_in_scenarios)}...", end=" ", flush=True)
        r = run_burn_in_scenario(s, T_BURNIN)
        results.append(r)
        print("done")

    metric_every = burn_in_scenarios[0]['metric_every']

    # Analyze convergence
    print("\nAnalyzing convergence...")
    convergence = analyze_convergence(results, metric_every)

    print("\nConvergence steps (first 500-step rolling mean within 2% of final):")
    for metric, steps in convergence.items():
        if steps:
            print(f"  {metric}: {steps} -> max={max(steps)}, mean={np.mean(steps):.0f}")
        else:
            print(f"  {metric}: no convergence detected")

    # Determine burn_in as max of all convergence steps + safety margin
    all_steps = []
    for steps in convergence.values():
        all_steps.extend(steps)

    if all_steps:
        max_step = max(all_steps)
        # Round up to nearest 500 and add 500 safety margin
        burn_in = ((max_step // 500) + 2) * 500
    else:
        # Default if no convergence detected
        burn_in = 2000

    # T should be burn_in + sufficient post-burn-in period
    # Use 4000 post-burn-in for statistics
    T = burn_in + 4000

    print(f"\nRecommended parameters:")
    print(f"  burn_in = {burn_in}")
    print(f"  T = {T}")

    # Save results
    output = {
        'T_burnin_analysis': T_BURNIN,
        'window': WINDOW,
        'tolerance': TOLERANCE,
        'final_window': FINAL_WINDOW,
        'convergence_steps': {k: v for k, v in convergence.items()},
        'recommended_burn_in': burn_in,
        'recommended_T': T,
    }

    Path('diagnostics').mkdir(exist_ok=True)
    with open('diagnostics/burn_in_c_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to diagnostics/burn_in_c_results.json")

    # Save time series for plotting
    timeseries = {
        'metric_every': metric_every,
        'T': T_BURNIN,
        'reps': [],
    }

    for i, r in enumerate(results):
        rep_data = {
            'rep': i,
            'hill_exponent': (np.nanmean(r['hill_exponent'], axis=0).tolist()
                              if r['hill_exponent'] is not None else []),
            'cong_capital_share': (r['cong_capital_share'].tolist()
                                   if r['cong_capital_share'] is not None else []),
            'mean_K': extract_mean_K(r['effective_members'], metric_every).tolist(),
        }
        timeseries['reps'].append(rep_data)

    with open('diagnostics/burn_in_c_timeseries.json', 'w') as f:
        json.dump(timeseries, f, indent=2)

    print(f"Saved time series to diagnostics/burn_in_c_timeseries.json")

    return output


if __name__ == '__main__':
    main()
