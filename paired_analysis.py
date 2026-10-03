#!/usr/bin/env python3
"""
Paired difference analysis for counterfactual experiments (§5).

Uses common random numbers (CRN) to compute proper inference.
Each experiment_id uses the same base seed across α values, enabling
matched-pair comparisons that eliminate between-replication variance.

Usage:
    python paired_analysis.py results_share_0.00.pkl results_share_0.10.pkl --output paired_diff.csv
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict


def load_results(filepath):
    """Load results pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def match_experiments(control_results, treatment_results):
    """
    Match experiments by experiment_id for paired comparison.

    Returns:
    --------
    matched_pairs : list of tuples
        [(control_result, treatment_result), ...] for matched pairs
    n_unmatched_control : int
        Number of control experiments without a match
    n_unmatched_treatment : int
        Number of treatment experiments without a match
    """
    # Index control results by experiment_id
    control_by_id = {r['experiment_id']: r for r in control_results}
    treatment_by_id = {r['experiment_id']: r for r in treatment_results}

    # Find matching pairs
    common_ids = set(control_by_id.keys()) & set(treatment_by_id.keys())

    matched_pairs = [
        (control_by_id[exp_id], treatment_by_id[exp_id])
        for exp_id in sorted(common_ids)
    ]

    n_unmatched_control = len(control_by_id) - len(common_ids)
    n_unmatched_treatment = len(treatment_by_id) - len(common_ids)

    return matched_pairs, n_unmatched_control, n_unmatched_treatment


def compute_paired_differences(matched_pairs, metric_name):
    """
    Compute differences in a metric across matched pairs.

    Parameters:
    -----------
    matched_pairs : list of tuples
        [(control_result, treatment_result), ...]
    metric_name : str
        Key in result dict (e.g., 'gini_coefficient', 'mean_members')

    Returns:
    --------
    differences : np.ndarray
        treatment - control for each pair (and each timestep if applicable)
    """
    differences = []
    for control, treatment in matched_pairs:
        ctrl_val = control.get(metric_name)
        treat_val = treatment.get(metric_name)

        if ctrl_val is None or treat_val is None:
            continue

        # Handle both scalar and array metrics
        ctrl_val = np.array(ctrl_val)
        treat_val = np.array(treat_val)

        # Ensure same shape
        if ctrl_val.shape != treat_val.shape:
            print(f"Warning: Shape mismatch for {metric_name}: "
                  f"control {ctrl_val.shape} vs treatment {treat_val.shape}")
            continue

        differences.append(treat_val - ctrl_val)

    return np.array(differences) if differences else None


def paired_t_test(differences, axis=0):
    """
    Compute paired t-test statistics.

    Parameters:
    -----------
    differences : np.ndarray
        Array of differences (first axis is replications)
    axis : int
        Axis along which to compute (default 0 for replications)

    Returns:
    --------
    mean_diff : float or ndarray
        Mean difference
    se_diff : float or ndarray
        Standard error of the mean difference
    t_stat : float or ndarray
        t-statistic
    n : int
        Number of pairs
    """
    n = differences.shape[axis]
    mean_diff = np.mean(differences, axis=axis)
    std_diff = np.std(differences, axis=axis, ddof=1)
    se_diff = std_diff / np.sqrt(n)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = mean_diff / se_diff
        t_stat = np.nan_to_num(t_stat, nan=0.0)

    return mean_diff, se_diff, t_stat, n


def analyze_key_metrics(matched_pairs, output_path=None):
    """
    Analyze key metrics using paired differences.

    Returns a DataFrame with summary statistics.
    """
    results = []

    # Key metrics to analyze
    metrics = [
        ('gini_avg', 'Gini coefficient'),
        ('mean_members', 'Mean conglomerate size'),
        ('num_cong', 'Number of conglomerates'),
    ]

    for metric_name, metric_label in metrics:
        diffs = compute_paired_differences(matched_pairs, metric_name)

        if diffs is None or len(diffs) == 0:
            print(f"Skipping {metric_name}: no valid pairs")
            continue

        # For time series, compute average over time first
        if diffs.ndim > 1:
            # Average over time (last axis)
            diffs_avg = np.mean(diffs, axis=-1)
        else:
            diffs_avg = diffs

        mean_diff, se_diff, t_stat, n = paired_t_test(diffs_avg)

        # 95% confidence interval
        ci_95 = 1.96 * se_diff

        results.append({
            'metric': metric_label,
            'mean_diff': mean_diff,
            'se': se_diff,
            'ci_lower': mean_diff - ci_95,
            'ci_upper': mean_diff + ci_95,
            't_stat': t_stat,
            'n_pairs': n,
            'significant_5pct': abs(t_stat) > 1.96
        })

        print(f"{metric_label}:")
        print(f"  Mean diff: {mean_diff:.6f} (SE: {se_diff:.6f})")
        print(f"  95% CI: [{mean_diff - ci_95:.6f}, {mean_diff + ci_95:.6f}]")
        print(f"  t-stat: {t_stat:.2f}, n={n}")
        print()

    df = pd.DataFrame(results)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return df


def analyze_time_series(matched_pairs, metric_name, output_path=None):
    """
    Analyze paired differences over time.

    Returns:
    --------
    df : pd.DataFrame
        Columns: step, mean_diff, se, ci_lower, ci_upper, t_stat
    """
    diffs = compute_paired_differences(matched_pairs, metric_name)

    if diffs is None or len(diffs) == 0:
        print(f"No valid pairs for {metric_name}")
        return None

    # diffs shape: (n_pairs, n_timesteps) or (n_pairs, n_markets, n_timesteps)
    # If 3D, average over markets first
    if diffs.ndim == 3:
        diffs = np.mean(diffs, axis=1)  # Average over markets

    n_steps = diffs.shape[1]
    results = []

    for step in range(n_steps):
        step_diffs = diffs[:, step]
        mean_diff, se_diff, t_stat, n = paired_t_test(step_diffs, axis=0)
        ci_95 = 1.96 * se_diff

        results.append({
            'step': step,
            'mean_diff': mean_diff,
            'se': se_diff,
            'ci_lower': mean_diff - ci_95,
            'ci_upper': mean_diff + ci_95,
            't_stat': t_stat,
            'n_pairs': n
        })

    df = pd.DataFrame(results)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Time series results saved to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Paired difference analysis for counterfactual experiments'
    )
    parser.add_argument('control', type=str,
                       help='Path to control (α=0) results pickle')
    parser.add_argument('treatment', type=str,
                       help='Path to treatment (α>0) results pickle')
    parser.add_argument('--output', type=str, default='paired_analysis.csv',
                       help='Output CSV path for summary statistics')
    parser.add_argument('--time_series', type=str, default=None,
                       help='If provided, output path for time series analysis')
    parser.add_argument('--metric', type=str, default='gini_avg',
                       help='Metric for time series analysis (default: gini_avg)')

    args = parser.parse_args()

    print(f"Loading control results from {args.control}...")
    control_data = load_results(args.control)

    print(f"Loading treatment results from {args.treatment}...")
    treatment_data = load_results(args.treatment)

    # Get individual experiment results
    control_results = control_data.get('individual_results', [])
    treatment_results = treatment_data.get('individual_results', [])

    if not control_results or not treatment_results:
        print("Error: No individual results found in pickle files.")
        print("Make sure results contain 'individual_results' key.")
        return

    print(f"\nControl experiments: {len(control_results)}")
    print(f"Treatment experiments: {len(treatment_results)}")

    # Match experiments
    matched_pairs, unmatched_ctrl, unmatched_treat = match_experiments(
        control_results, treatment_results
    )

    print(f"Matched pairs: {len(matched_pairs)}")
    if unmatched_ctrl > 0:
        print(f"Warning: {unmatched_ctrl} control experiments without match")
    if unmatched_treat > 0:
        print(f"Warning: {unmatched_treat} treatment experiments without match")

    if len(matched_pairs) == 0:
        print("Error: No matched pairs found.")
        return

    # Analyze key metrics
    print("\n=== Paired Difference Analysis ===\n")
    analyze_key_metrics(matched_pairs, args.output)

    # Time series analysis if requested
    if args.time_series:
        print(f"\n=== Time Series Analysis: {args.metric} ===\n")
        analyze_time_series(matched_pairs, args.metric, args.time_series)


if __name__ == '__main__':
    main()
