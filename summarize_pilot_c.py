#!/usr/bin/env python3
"""
Phase C pilot result summarization.

Loads pilot_c/results/scenario_*.pkl.gz files and produces:
- pilot_c/tidy.csv: one row per scenario with scalar metrics
- pilot_c/medians.csv: medians and 25-75% bands grouped by cell

Statistics computed over t >= burn_in.
"""
import gzip
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def extract_post_burnin_metrics(result):
    """
    Extract summary statistics from result for t >= burn_in.

    Returns dict of scalar metrics.
    """
    scenario = result['scenario']
    burn_in = scenario['burn_in']
    metric_every = scenario['metric_every']
    T = scenario['T']

    # Number of metric observations post burn-in
    n_obs = (T - burn_in) // metric_every

    metrics = {
        # Scenario identifiers
        'scenario_id': scenario['scenario_id'],
        'cell_id': scenario['cell_id'],
        'block': scenario['block'],
        'log_family': scenario['log_family'],
        'cost_type': scenario['cost_type'],
        'alpha': scenario['alpha'],
        'rep': scenario['rep'],
        'seed': scenario['seed'],
        'sharing_rule': scenario['sharing_rule'],
        'lookback': scenario['lookback'],
        'cross_corr': scenario['cross_corr'],
        'alpha_endogenous': scenario.get('alpha_endogenous', False),
        'decision_rule': scenario.get('decision_rule', 'replay'),  # C17
        # Runtime
        'elapsed_seconds': result['elapsed_seconds'],
        'ms_per_step': result['ms_per_step'],
    }

    # Add cost_multiplier if present (cost-level block)
    if 'cost_multiplier' in scenario:
        metrics['cost_multiplier'] = scenario['cost_multiplier']
    else:
        metrics['cost_multiplier'] = 1.0

    # Add floor_c for summary report
    metrics['floor_c'] = scenario.get('floor_c', 0.0)

    # Summary scalars from model
    summary = result.get('summary', {})
    metrics['K_median'] = summary.get('K_median', np.nan)
    metrics['K_mean'] = summary.get('K_mean', np.nan)  # C17
    metrics['K_eff_over_K_median'] = summary.get('K_eff_over_K_median', np.nan)
    metrics['floor_hit_rate_standalone'] = summary.get('floor_hit_rate_standalone', np.nan)
    metrics['floor_hit_rate_member'] = summary.get('floor_hit_rate_member', np.nan)
    metrics['mergers_per_period'] = summary.get('mergers_per_period', np.nan)
    metrics['proposals_per_period'] = summary.get('proposals_per_period', np.nan)
    metrics['exits_per_period'] = summary.get('exits_per_period', np.nan)
    metrics['growth_gap_median'] = summary.get('growth_gap_median', np.nan)  # C17

    # Per-type acceptance rates (C17)
    metrics['acceptance_rate_ss'] = summary.get('acceptance_rate_ss', np.nan)
    metrics['acceptance_rate_sc'] = summary.get('acceptance_rate_sc', np.nan)
    metrics['acceptance_rate_cc'] = summary.get('acceptance_rate_cc', np.nan)

    # Acceptance rate = mergers / proposals
    mergers = summary.get('mergers_per_period', np.nan)
    proposals = summary.get('proposals_per_period', np.nan)
    if proposals and proposals > 0:
        metrics['acceptance_rate'] = mergers / proposals
    else:
        metrics['acceptance_rate'] = np.nan

    # Time series metrics: extract post-burn-in window
    # Hill exponent: array of shape (markets, n_observations)
    hill = result.get('hill_exponent')
    if hill is not None and len(hill) > 0:
        # Skip burn-in observations
        burn_in_obs = burn_in // metric_every
        hill_post = hill[:, burn_in_obs:] if hill.ndim == 2 else hill[burn_in_obs:]
        # Median across markets and time
        metrics['hill_exponent_median'] = np.nanmedian(hill_post)
        metrics['hill_exponent_p25'] = np.nanpercentile(hill_post, 25)
        metrics['hill_exponent_p75'] = np.nanpercentile(hill_post, 75)
    else:
        metrics['hill_exponent_median'] = np.nan
        metrics['hill_exponent_p25'] = np.nan
        metrics['hill_exponent_p75'] = np.nan

    # HHI within-market: array of shape (markets, n_observations)
    # Take mean across markets, then extract post-burn-in
    hhi_within = result.get('hhi_within')
    if hhi_within is not None and len(hhi_within) > 0:
        burn_in_obs = burn_in // metric_every
        # Handle both 2D (markets, obs) and 1D (obs) shapes
        if hhi_within.ndim == 2:
            hhi_mean = np.nanmean(hhi_within, axis=0)  # Mean across markets
            hhi_post = hhi_mean[burn_in_obs:]
        else:
            hhi_post = hhi_within[burn_in_obs:]
        metrics['hhi_within_median'] = np.nanmedian(hhi_post)
        metrics['hhi_within_p25'] = np.nanpercentile(hhi_post, 25)
        metrics['hhi_within_p75'] = np.nanpercentile(hhi_post, 75)
    else:
        metrics['hhi_within_median'] = np.nan
        metrics['hhi_within_p25'] = np.nan
        metrics['hhi_within_p75'] = np.nan

    # HHI aggregate
    hhi_agg = result.get('hhi_aggregate')
    if hhi_agg is not None and len(hhi_agg) > 0:
        burn_in_obs = burn_in // metric_every
        hhi_post = hhi_agg[burn_in_obs:]
        metrics['hhi_aggregate_median'] = np.nanmedian(hhi_post)
        metrics['hhi_aggregate_p25'] = np.nanpercentile(hhi_post, 25)
        metrics['hhi_aggregate_p75'] = np.nanpercentile(hhi_post, 75)
    else:
        metrics['hhi_aggregate_median'] = np.nan
        metrics['hhi_aggregate_p25'] = np.nan
        metrics['hhi_aggregate_p75'] = np.nan

    # Top-10 aggregate share
    top10 = result.get('top10pct_aggregate')
    if top10 is not None and len(top10) > 0:
        burn_in_obs = burn_in // metric_every
        top10_post = top10[burn_in_obs:]
        metrics['top10pct_aggregate_median'] = np.nanmedian(top10_post)
        metrics['top10pct_aggregate_p25'] = np.nanpercentile(top10_post, 25)
        metrics['top10pct_aggregate_p75'] = np.nanpercentile(top10_post, 75)
    else:
        metrics['top10pct_aggregate_median'] = np.nan
        metrics['top10pct_aggregate_p25'] = np.nan
        metrics['top10pct_aggregate_p75'] = np.nan

    # Conglomerate capital share
    ccs = result.get('cong_capital_share')
    if ccs is not None and len(ccs) > 0:
        burn_in_obs = burn_in // metric_every
        ccs_post = ccs[burn_in_obs:]
        metrics['cong_capital_share_median'] = np.nanmedian(ccs_post)
        metrics['cong_capital_share_p25'] = np.nanpercentile(ccs_post, 25)
        metrics['cong_capital_share_p75'] = np.nanpercentile(ccs_post, 75)
    else:
        metrics['cong_capital_share_median'] = np.nan
        metrics['cong_capital_share_p25'] = np.nan
        metrics['cong_capital_share_p75'] = np.nan

    # K and K_eff from effective_members
    eff_members = result.get('effective_members')
    if eff_members:
        K_values = []
        K_eff_values = []
        for step_val, congs in eff_members:
            if step_val >= burn_in and congs:
                for cid, K, K_eff in congs:
                    K_values.append(K)
                    K_eff_values.append(K_eff)
        if K_values:
            metrics['K_post_burnin_median'] = np.median(K_values)
            metrics['K_post_burnin_p25'] = np.percentile(K_values, 25)
            metrics['K_post_burnin_p75'] = np.percentile(K_values, 75)
            metrics['K_eff_post_burnin_median'] = np.median(K_eff_values)
            metrics['K_eff_post_burnin_p25'] = np.percentile(K_eff_values, 25)
            metrics['K_eff_post_burnin_p75'] = np.percentile(K_eff_values, 75)
        else:
            metrics['K_post_burnin_median'] = np.nan
            metrics['K_post_burnin_p25'] = np.nan
            metrics['K_post_burnin_p75'] = np.nan
            metrics['K_eff_post_burnin_median'] = np.nan
            metrics['K_eff_post_burnin_p25'] = np.nan
            metrics['K_eff_post_burnin_p75'] = np.nan
    else:
        metrics['K_post_burnin_median'] = np.nan
        metrics['K_post_burnin_p25'] = np.nan
        metrics['K_post_burnin_p75'] = np.nan
        metrics['K_eff_post_burnin_median'] = np.nan
        metrics['K_eff_post_burnin_p25'] = np.nan
        metrics['K_eff_post_burnin_p75'] = np.nan

    # Endogenous alpha: final adopted alpha
    if scenario.get('alpha_endogenous', False):
        alpha_hist = result.get('alpha_history')
        if alpha_hist is not None:
            # alpha_history is a 2D array: (max_conglomerates, n_metric_steps)
            # Get final alphas for each conglomerate that was active
            final_alphas = []
            if isinstance(alpha_hist, np.ndarray) and alpha_hist.ndim == 2:
                # Find conglomerates that had non-zero alpha at some point
                for cid in range(alpha_hist.shape[0]):
                    # Get last non-zero alpha for this conglomerate
                    cid_alphas = alpha_hist[cid, :]
                    valid_alphas = cid_alphas[cid_alphas > 0]
                    if len(valid_alphas) > 0:
                        final_alphas.append(valid_alphas[-1])
            if final_alphas:
                metrics['alpha_adopted_median'] = np.median(final_alphas)
                metrics['alpha_adopted_mean'] = np.mean(final_alphas)
                metrics['alpha_adopted_std'] = np.std(final_alphas)
            else:
                metrics['alpha_adopted_median'] = np.nan
                metrics['alpha_adopted_mean'] = np.nan
                metrics['alpha_adopted_std'] = np.nan
        else:
            metrics['alpha_adopted_median'] = np.nan
            metrics['alpha_adopted_mean'] = np.nan
            metrics['alpha_adopted_std'] = np.nan
    else:
        metrics['alpha_adopted_median'] = np.nan
        metrics['alpha_adopted_mean'] = np.nan
        metrics['alpha_adopted_std'] = np.nan

    # Assortativity: use pre-computed assort_iqr from model (C22c)
    metrics["assort_iqr"] = summary.get("assort_iqr", np.nan)

    return metrics


def compute_assortativity_ratio(market_iqr, firm_conglom, n_random_samples=100):
    """
    Compute assortativity ratio: SD of member IQR / SD of random K-subset.

    Parameters:
    -----------
    market_iqr : array, shape (n_markets,)
        Per-market IQR of growth rates
    firm_conglom : array, shape (n_firms,)
        Conglomerate ID for each firm (-1 for standalone)

    Returns:
    --------
    dict with:
        - ratio_median: median ratio across conglomerates
        - ratio_mean: mean ratio
        - n_conglom: number of multi-member conglomerates
    """
    # Build mapping from firm index to market IQR
    n_firms = len(firm_conglom)
    n_markets = len(market_iqr)

    if n_markets == 0 or n_firms == 0:
        return {'ratio_median': np.nan, 'ratio_mean': np.nan, 'n_conglom': 0}

    # Map firms to markets (assuming firms_per_market = n_firms / n_markets)
    firms_per_market = n_firms // n_markets
    firm_iqr = np.repeat(market_iqr, firms_per_market)

    # Find multi-member conglomerates
    unique_conglom = np.unique(firm_conglom[firm_conglom >= 0])
    ratios = []

    for cid in unique_conglom:
        member_mask = firm_conglom == cid
        K = np.sum(member_mask)
        if K < 2:
            continue

        # SD of member IQR
        member_iqrs = firm_iqr[member_mask]
        sd_member = np.std(member_iqrs, ddof=1)

        # SD of random K-subsets
        random_sds = []
        for _ in range(n_random_samples):
            random_idx = np.random.choice(n_firms, size=K, replace=False)
            random_iqrs = firm_iqr[random_idx]
            random_sds.append(np.std(random_iqrs, ddof=1))

        sd_random = np.mean(random_sds)

        if sd_random > 0:
            ratios.append(sd_member / sd_random)

    if len(ratios) == 0:
        return {'ratio_median': np.nan, 'ratio_mean': np.nan, 'n_conglom': 0}

    return {
        'ratio_median': np.median(ratios),
        'ratio_mean': np.mean(ratios),
        'n_conglom': len(ratios),
    }


def load_results(results_dir='pilot_c/results'):
    """Load all result files (.pkl.gz or .pkl)."""
    results_path = Path(results_dir)

    # Find all result files (both compressed and uncompressed)
    gz_files = set(results_path.glob('scenario_*.pkl.gz'))
    pkl_files = set(results_path.glob('scenario_*.pkl'))

    # Prefer .pkl.gz if both exist for same scenario
    gz_bases = {f.stem.replace('.pkl', '') for f in gz_files}  # scenario_XXXX
    pkl_only = [f for f in pkl_files if f.stem not in gz_bases]

    all_files = sorted(list(gz_files) + pkl_only, key=lambda f: f.name)

    results = []
    for result_file in all_files:
        try:
            if result_file.suffix == '.gz':
                with gzip.open(result_file, 'rb') as f:
                    result = pickle.load(f)
            else:
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
            continue

    return results


def create_tidy_df(results):
    """Create tidy DataFrame with one row per scenario."""
    rows = []
    for result in results:
        metrics = extract_post_burnin_metrics(result)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    return df


def create_medians_df(tidy_df):
    """
    Create medians DataFrame grouped by cell.

    Groups by (block, log_family, cost_type, alpha, sharing_rule, lookback,
               cross_corr, alpha_endogenous, cost_multiplier)
    and computes median and 25-75% bands across reps.
    """
    group_cols = ['block', 'log_family', 'cost_type', 'alpha', 'sharing_rule',
                  'lookback', 'cross_corr', 'alpha_endogenous', 'cost_multiplier',
                  'decision_rule']  # C17

    # Numeric columns to aggregate
    value_cols = [
        'K_median', 'K_mean', 'K_eff_over_K_median',  # C17: K_mean added
        'K_post_burnin_median', 'K_eff_post_burnin_median',
        'floor_hit_rate_standalone', 'floor_hit_rate_member',
        'hill_exponent_median', 'hhi_within_median', 'hhi_aggregate_median',
        'top10pct_aggregate_median', 'cong_capital_share_median',
        'mergers_per_period', 'proposals_per_period', 'exits_per_period',
        'acceptance_rate', 'assort_iqr',
        'acceptance_rate_ss', 'acceptance_rate_sc', 'acceptance_rate_cc',  # C17
        'growth_gap_median',  # C17
        'elapsed_seconds', 'ms_per_step',
        'alpha_adopted_median', 'alpha_adopted_mean',
    ]

    agg_dict = {}
    for col in value_cols:
        if col in tidy_df.columns:
            agg_dict[col] = ['median', lambda x: np.percentile(x.dropna(), 25) if len(x.dropna()) > 0 else np.nan,
                            lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) > 0 else np.nan]

    medians_df = tidy_df.groupby(group_cols, dropna=False).agg(agg_dict)

    # Flatten column names
    medians_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                          for col in medians_df.columns.values]
    # Rename lambda columns to p25, p75
    medians_df.columns = [c.replace('<lambda_0>', 'p25').replace('<lambda_1>', 'p75')
                          for c in medians_df.columns]

    medians_df = medians_df.reset_index()

    # Also add rep count
    rep_counts = tidy_df.groupby(group_cols, dropna=False).size().reset_index(name='n_reps')
    medians_df = medians_df.merge(rep_counts, on=group_cols)

    return medians_df


def main():
    """Load results and create summary CSVs."""
    print("Phase C Pilot Summarization")
    print("=" * 50)

    # Load results
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} results")

    if len(results) == 0:
        print("No results found. Exiting.")
        return

    # Create tidy DataFrame
    print("Creating tidy DataFrame...")
    tidy_df = create_tidy_df(results)

    # Save tidy.csv
    tidy_path = 'pilot_c/tidy.csv'
    tidy_df.to_csv(tidy_path, index=False)
    print(f"Saved {tidy_path} ({len(tidy_df)} rows)")

    # Create medians DataFrame
    print("Creating medians DataFrame...")
    medians_df = create_medians_df(tidy_df)

    # Save medians.csv
    medians_path = 'pilot_c/medians.csv'
    medians_df.to_csv(medians_path, index=False)
    print(f"Saved {medians_path} ({len(medians_df)} rows)")

    # Summary statistics
    print("\nSummary:")
    print(f"  Blocks: {tidy_df['block'].unique().tolist()}")
    print(f"  Families: {tidy_df['log_family'].unique().tolist()}")
    print(f"  Cost types: {tidy_df['cost_type'].unique().tolist()}")
    print(f"  Alpha values: {sorted(tidy_df['alpha'].unique())}")
    print(f"  Total runtime: {tidy_df['elapsed_seconds'].sum():.0f}s "
          f"({tidy_df['elapsed_seconds'].sum() / 3600:.1f} CPU-hours)")
    print(f"  Mean ms/step: {tidy_df['ms_per_step'].mean():.2f}")


if __name__ == '__main__':
    main()
