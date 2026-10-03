#!/usr/bin/env python3
"""
Summarize counterfactual results into tidy CSV files.

Reads individual pickle files (alpha_X.XX_rep_NNN.pkl) and produces:
- tidy.csv: Long format with one row per (alpha, rep)
- medians.csv: Median values across replications for each alpha
- paired_diffs.csv: Paired differences between alpha values (common random numbers)

Usage:
    python summarize_results.py --results_dir results/test_dryrun
    python summarize_results.py --results_dir results/scenario_name --output_dir summaries/
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all pickle files from results directory."""
    results_dir = Path(results_dir)
    results = []

    for pkl_file in sorted(results_dir.glob('alpha_*.pkl')):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {pkl_file}: {e}")

    return results


def build_summary_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build summary dict from a single experiment result.

    Extracts key scalar metrics for CSV output.
    """
    summary = {
        'alpha': result['alpha'],
        'rep': result['rep'],
        'seed': result['seed'],
    }

    # Final values (last timestep)
    if 'mean_members' in result:
        summary['mean_members_final'] = result['mean_members'][-1]

    if 'num_cong' in result:
        summary['num_cong_final'] = result['num_cong'][-1]

    if 'gini_coefficient' in result:
        # gini_coefficient is averaged over markets, shape (steps,)
        summary['gini_final'] = result['gini_coefficient'][-1]
        summary['gini_mean'] = np.mean(result['gini_coefficient'])

    # Market share quantiles at final step
    if 'market_share_quantiles' in result:
        # Shape: (7_quantiles, steps) - quantiles are [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
        qs = result['market_share_quantiles']
        summary['share_p50_final'] = qs[2, -1]  # median
        summary['share_p90_final'] = qs[4, -1]  # 90th percentile
        summary['share_max_final'] = qs[6, -1]  # max

    # Merger and exit totals
    if 'mergers_per_period' in result:
        summary['total_mergers'] = np.sum(result['mergers_per_period'])
        summary['mergers_per_period_mean'] = np.mean(result['mergers_per_period'])

    if 'exits_per_period' in result:
        summary['total_exits'] = np.sum(result['exits_per_period'])
        summary['exits_per_period_mean'] = np.mean(result['exits_per_period'])

    if 'proposals_per_period' in result:
        summary['total_proposals'] = np.sum(result['proposals_per_period'])

    # Panel polynomial coefficients (size-market share relationship)
    if 'panel_poly_estimates' in result:
        coeffs = result['panel_poly_estimates']
        if len(coeffs) == 3:
            summary['poly_beta2'] = coeffs[0]  # quadratic
            summary['poly_beta1'] = coeffs[1]  # linear
            summary['poly_beta0'] = coeffs[2]  # intercept

    # Hyperparameters (selected)
    if 'hyperparameters' in result and result['hyperparameters']:
        hp = result['hyperparameters']
        summary['growth_process'] = hp.get('growth_process', 'normal_net')
        summary['sharing_rule'] = hp.get('sharing_rule', 'equal')
        summary['rho'] = hp.get('rho', 0.0)
        summary['cross_corr'] = hp.get('cross_corr', 0.0)
        summary['markets'] = hp.get('markets', 100)
        summary['firms_per_market'] = hp.get('firms_per_market', 100)
        summary['steps'] = hp.get('steps', 10000)

    return summary


def create_tidy_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create tidy (long format) DataFrame from results."""
    summaries = [build_summary_dict(r) for r in results]
    df = pd.DataFrame(summaries)

    # Sort by alpha, then rep
    df = df.sort_values(['alpha', 'rep']).reset_index(drop=True)

    return df


def create_medians_df(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create medians DataFrame: median values across replications for each alpha.
    """
    # Numeric columns only (exclude categorical hyperparameters and alpha itself)
    numeric_cols = [c for c in tidy_df.select_dtypes(include=[np.number]).columns
                    if c != 'alpha']

    # Group by alpha and compute medians
    medians = tidy_df.groupby('alpha')[numeric_cols].median()

    # Add count of replications
    medians['n_reps'] = tidy_df.groupby('alpha').size()

    return medians.reset_index()


def create_paired_diffs_df(tidy_df: pd.DataFrame, baseline_alpha: float = 0.0) -> pd.DataFrame:
    """
    Create paired differences DataFrame.

    For each rep, compute difference between each alpha and baseline_alpha.
    This exploits common random numbers (same rep = same seed across alphas).
    """
    # Get baseline data
    baseline = tidy_df[tidy_df['alpha'] == baseline_alpha].set_index('rep')

    # Get all other alphas
    other_alphas = tidy_df[tidy_df['alpha'] != baseline_alpha]

    # Numeric columns for differencing (exclude identifiers and categoricals)
    exclude_cols = ['alpha', 'rep', 'seed', 'growth_process', 'sharing_rule',
                    'markets', 'firms_per_market', 'steps']
    numeric_cols = [c for c in tidy_df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols]

    paired_rows = []
    for _, row in other_alphas.iterrows():
        rep = row['rep']
        alpha = row['alpha']

        if rep not in baseline.index:
            continue

        base_row = baseline.loc[rep]

        diff_row = {
            'alpha': alpha,
            'baseline_alpha': baseline_alpha,
            'rep': rep,
            'seed': row['seed'],
        }

        # Compute differences for each numeric column
        for col in numeric_cols:
            if col in row and col in base_row:
                diff_row[f'{col}_diff'] = row[col] - base_row[col]
                diff_row[f'{col}_treatment'] = row[col]
                diff_row[f'{col}_baseline'] = base_row[col]

        paired_rows.append(diff_row)

    df = pd.DataFrame(paired_rows)

    if len(df) > 0:
        df = df.sort_values(['alpha', 'rep']).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Summarize counterfactual results into CSV files'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing alpha_*.pkl files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for CSV files (default: same as results_dir)')
    parser.add_argument('--baseline_alpha', type=float, default=0.0,
                        help='Baseline alpha for paired differences (default: 0.0)')

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} experiment results")

    # Create tidy DataFrame
    print("Creating tidy.csv...")
    tidy_df = create_tidy_df(results)
    tidy_path = output_dir / 'tidy.csv'
    tidy_df.to_csv(tidy_path, index=False)
    print(f"  Saved {tidy_path} ({len(tidy_df)} rows)")

    # Create medians DataFrame
    print("Creating medians.csv...")
    medians_df = create_medians_df(tidy_df)
    medians_path = output_dir / 'medians.csv'
    medians_df.to_csv(medians_path, index=False)
    print(f"  Saved {medians_path} ({len(medians_df)} rows)")

    # Create paired differences DataFrame
    print(f"Creating paired_diffs.csv (baseline α={args.baseline_alpha})...")
    paired_df = create_paired_diffs_df(tidy_df, baseline_alpha=args.baseline_alpha)
    paired_path = output_dir / 'paired_diffs.csv'
    paired_df.to_csv(paired_path, index=False)
    print(f"  Saved {paired_path} ({len(paired_df)} rows)")

    # Print summary
    print("\n=== Summary ===")
    print(f"Alpha values: {sorted(tidy_df['alpha'].unique())}")
    print(f"Replications per alpha: {tidy_df.groupby('alpha').size().to_dict()}")

    if 'gini_final' in medians_df.columns:
        print("\nMedian Gini (final) by alpha:")
        for _, row in medians_df.iterrows():
            print(f"  α={row['alpha']:.2f}: {row['gini_final']:.4f}")


if __name__ == '__main__':
    main()
