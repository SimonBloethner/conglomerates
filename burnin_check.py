#!/usr/bin/env python3
"""
Burn-in check analysis (§6).

Verifies that the simulation has reached stationarity before the analysis window.
Uses rolling window statistics to detect convergence.

Methods:
1. Visual diagnostics: Plot time series of key metrics
2. Geweke test: Compare means of early vs late portions
3. Autocorrelation decay: Check if autocorrelation drops to noise level

Usage:
    python burnin_check.py results.pkl --metric gini_avg --output burnin_report.csv
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy import stats


def load_results(filepath):
    """Load results pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def geweke_test(series, first_frac=0.1, last_frac=0.5, z_threshold=1.96):
    """
    Geweke convergence diagnostic.

    Compares the mean of the first portion vs last portion of the series.
    Under stationarity, the z-score should be small.

    Parameters:
    -----------
    series : np.ndarray
        Time series to test
    first_frac : float
        Fraction of series to use as "early" window
    last_frac : float
        Fraction of series to use as "late" window
    z_threshold : float
        Z-score threshold for declaring non-stationarity

    Returns:
    --------
    z_score : float
        Test statistic
    p_value : float
        Two-tailed p-value
    converged : bool
        True if z_score < z_threshold
    """
    n = len(series)
    n_first = int(n * first_frac)
    n_last = int(n * last_frac)

    first_window = series[:n_first]
    last_window = series[-n_last:]

    mean_first = np.mean(first_window)
    mean_last = np.mean(last_window)

    # Spectral density at frequency 0 (approximated by variance / effective sample size)
    # Simplified: use variance directly
    var_first = np.var(first_window, ddof=1) / n_first
    var_last = np.var(last_window, ddof=1) / n_last

    # Z-score for difference in means
    se_diff = np.sqrt(var_first + var_last)
    if se_diff < 1e-10:
        z_score = 0.0
    else:
        z_score = (mean_first - mean_last) / se_diff

    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    converged = abs(z_score) < z_threshold

    return z_score, p_value, converged


def effective_sample_size(series, max_lag=None):
    """
    Estimate effective sample size accounting for autocorrelation.

    Parameters:
    -----------
    series : np.ndarray
        Time series
    max_lag : int, optional
        Maximum lag to consider (default: n/2)

    Returns:
    --------
    n_eff : float
        Effective sample size
    autocorr_sum : float
        Sum of autocorrelations (diagnostic)
    """
    n = len(series)
    if max_lag is None:
        max_lag = n // 2

    # Center the series
    x = series - np.mean(series)

    # Compute autocorrelation
    acf = np.correlate(x, x, mode='full')[n-1:]
    acf = acf / acf[0]

    # Sum autocorrelations until they become small
    autocorr_sum = 0.0
    for lag in range(1, min(max_lag, n)):
        if acf[lag] < 0.05:  # Stop when autocorr becomes negligible
            break
        autocorr_sum += acf[lag]

    # Effective sample size
    n_eff = n / (1 + 2 * autocorr_sum)

    return n_eff, autocorr_sum


def analyze_burnin(series, metric_name, burnin_frac=0.1):
    """
    Comprehensive burn-in analysis for a single time series.

    Parameters:
    -----------
    series : np.ndarray
        Time series to analyze (full series including burn-in)
    metric_name : str
        Name of the metric for reporting
    burnin_frac : float
        Assumed burn-in fraction (will test if this is sufficient)

    Returns:
    --------
    results : dict
        Analysis results
    """
    n = len(series)
    burnin_end = int(n * burnin_frac)

    # Analysis on post-burn-in portion
    post_burnin = series[burnin_end:]

    # Geweke test
    z_score, p_value, converged = geweke_test(post_burnin)

    # Effective sample size
    n_eff, autocorr_sum = effective_sample_size(post_burnin)

    # Rolling mean stability (check if mean is stable in late portion)
    late_portion = post_burnin[-len(post_burnin)//2:]
    early_portion = post_burnin[:len(post_burnin)//2]
    mean_drift = abs(np.mean(late_portion) - np.mean(early_portion))
    mean_overall = np.mean(post_burnin)
    drift_pct = 100 * mean_drift / max(abs(mean_overall), 1e-10)

    # Variance stability
    var_late = np.var(late_portion)
    var_early = np.var(early_portion)
    var_ratio = var_late / max(var_early, 1e-10)

    results = {
        'metric': metric_name,
        'n_total': n,
        'burnin_steps': burnin_end,
        'n_post_burnin': len(post_burnin),
        'geweke_z': z_score,
        'geweke_p': p_value,
        'geweke_converged': converged,
        'n_eff': n_eff,
        'autocorr_sum': autocorr_sum,
        'eff_ratio': n_eff / len(post_burnin),
        'mean_drift_pct': drift_pct,
        'var_ratio': var_ratio,
        'mean_post_burnin': mean_overall,
        'std_post_burnin': np.std(post_burnin)
    }

    return results


def recommend_burnin(series, target_eff_ratio=0.5, max_burnin_frac=0.5):
    """
    Recommend optimal burn-in period.

    Finds the smallest burn-in such that:
    1. Geweke test passes
    2. Effective sample size ratio > target_eff_ratio

    Parameters:
    -----------
    series : np.ndarray
        Time series
    target_eff_ratio : float
        Target effective/actual sample size ratio
    max_burnin_frac : float
        Maximum fraction to consider for burn-in

    Returns:
    --------
    recommended_burnin : int
        Recommended burn-in period
    analysis : list of dict
        Analysis results for each burn-in fraction tested
    """
    n = len(series)
    fractions_to_test = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    fractions_to_test = [f for f in fractions_to_test if f <= max_burnin_frac]

    analysis = []
    recommended_burnin = int(n * fractions_to_test[0])

    for frac in fractions_to_test:
        result = analyze_burnin(series, f'frac_{frac:.2f}', burnin_frac=frac)
        result['burnin_frac'] = frac
        analysis.append(result)

        # Check if this burn-in is sufficient
        if result['geweke_converged'] and result['eff_ratio'] >= target_eff_ratio:
            recommended_burnin = result['burnin_steps']
            break

    return recommended_burnin, analysis


def analyze_experiment_results(results_dict, metric_name='gini_avg', output_path=None):
    """
    Analyze burn-in for experiment results.

    Parameters:
    -----------
    results_dict : dict
        Results from parallel_counterfactuals
    metric_name : str
        Metric to analyze (e.g., 'gini_avg', 'mean_members_avg')
    output_path : str, optional
        Path to save results CSV

    Returns:
    --------
    df : pd.DataFrame
        Analysis results
    """
    series = results_dict.get(metric_name)

    if series is None:
        print(f"Metric {metric_name} not found in results")
        return None

    # Handle multi-dimensional arrays (e.g., per-market metrics)
    if series.ndim > 1:
        # Average over non-time dimensions
        series = np.mean(series, axis=tuple(range(series.ndim - 1)))

    print(f"\nAnalyzing: {metric_name}")
    print(f"Series length: {len(series)} steps")

    # Default burn-in analysis
    results_10pct = analyze_burnin(series, metric_name, burnin_frac=0.10)

    print(f"\n=== Burn-in Analysis (10% = {results_10pct['burnin_steps']} steps) ===")
    print(f"Geweke test: z = {results_10pct['geweke_z']:.3f}, p = {results_10pct['geweke_p']:.4f}")
    print(f"Converged: {results_10pct['geweke_converged']}")
    print(f"Effective sample size: {results_10pct['n_eff']:.1f} ({100*results_10pct['eff_ratio']:.1f}% of actual)")
    print(f"Mean drift: {results_10pct['mean_drift_pct']:.2f}%")
    print(f"Variance ratio (late/early): {results_10pct['var_ratio']:.3f}")

    # Recommendation
    recommended, analysis_list = recommend_burnin(series)
    print(f"\nRecommended burn-in: {recommended} steps ({100*recommended/len(series):.1f}%)")

    # Create DataFrame
    df = pd.DataFrame(analysis_list)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Burn-in check analysis for simulation results'
    )
    parser.add_argument('results', type=str,
                       help='Path to results pickle file')
    parser.add_argument('--metric', type=str, default='gini_avg',
                       help='Metric to analyze (default: gini_avg)')
    parser.add_argument('--output', type=str, default='burnin_report.csv',
                       help='Output CSV path')
    parser.add_argument('--all_metrics', action='store_true',
                       help='Analyze all available time series metrics')

    args = parser.parse_args()

    print(f"Loading results from {args.results}...")
    results = load_results(args.results)

    if args.all_metrics:
        # Analyze all time series metrics
        time_series_metrics = ['gini_avg', 'mean_members_avg', 'num_cong_avg',
                               'mergers_per_period_avg', 'exits_per_period_avg']
        all_results = []

        for metric in time_series_metrics:
            if metric in results:
                df = analyze_experiment_results(results, metric)
                if df is not None:
                    df['metric_analyzed'] = metric
                    all_results.append(df)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(args.output, index=False)
            print(f"\nCombined results saved to {args.output}")
    else:
        analyze_experiment_results(results, args.metric, args.output)


if __name__ == '__main__':
    main()
