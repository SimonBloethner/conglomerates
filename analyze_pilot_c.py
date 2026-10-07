#!/usr/bin/env python3
"""
Phase C pilot analysis.

Loads pilot_c/tidy.csv and produces diagnostics/pilot_c_summary.md with:
- K and K_eff versus K* by family and cost
- Hill exponent at α=0 versus 1/(1-c); exponent versus α by family
- Floor-hit rate by status (member vs standalone) by family and α
- HHI (within and aggregate) and top-10 share versus α by family
- Equal-split check versus main normal block
- Lookback and correlation rows versus main laplace/power_law cell
- Endogenous α histogram and spread analysis
- Runtime statistics

Statistics: medians and 25-75% bands across replications.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_tidy_csv(path='pilot_c/tidy.csv'):
    """Load tidy.csv with scenario results."""
    return pd.read_csv(path)


def load_benchmarks(path='analytics/benchmarks.csv'):
    """Load analytical benchmarks with K*."""
    return pd.read_csv(path)


def agg_with_bands(df, group_cols, value_col):
    """
    Aggregate with median and 25-75% bands.

    Returns DataFrame with columns: {value_col}_median, {value_col}_p25, {value_col}_p75
    """
    result = df.groupby(group_cols, dropna=False)[value_col].agg([
        ('median', 'median'),
        ('p25', lambda x: np.nanpercentile(x, 25)),
        ('p75', lambda x: np.nanpercentile(x, 75)),
        ('n', 'count'),
    ]).reset_index()
    result.columns = group_cols + [f'{value_col}_median', f'{value_col}_p25',
                                   f'{value_col}_p75', 'n_reps']
    return result


def fmt_band(median, p25, p75):
    """Format median [p25, p75] band."""
    if pd.isna(median):
        return "—"
    return f"{median:.3f} [{p25:.3f}, {p75:.3f}]"


def generate_k_vs_kstar_table(df, benchmarks):
    """Generate table: K and K_eff versus K* by family/cost/alpha, with acceptance rate."""
    # Filter main block
    main = df[df['block'] == 'main'].copy()

    # Aggregate K by family, cost, alpha
    k_agg = agg_with_bands(main, ['log_family', 'cost_type', 'alpha'], 'K_post_burnin_median')
    k_eff_agg = agg_with_bands(main, ['log_family', 'cost_type', 'alpha'], 'K_eff_post_burnin_median')

    # Aggregate acceptance rate
    if 'acceptance_rate' in main.columns:
        acc_agg = agg_with_bands(main, ['log_family', 'cost_type', 'alpha'], 'acceptance_rate')
    else:
        acc_agg = None

    # Merge
    merged = k_agg.merge(k_eff_agg, on=['log_family', 'cost_type', 'alpha', 'n_reps'],
                         suffixes=('_K', '_K_eff'))

    if acc_agg is not None:
        merged = merged.merge(acc_agg, on=['log_family', 'cost_type', 'alpha', 'n_reps'])

    # Merge with benchmarks for K*
    if len(benchmarks) > 0:
        merged = merged.merge(
            benchmarks[['family', 'cost_type', 'alpha', 'K_star']],
            left_on=['log_family', 'cost_type', 'alpha'],
            right_on=['family', 'cost_type', 'alpha'],
            how='left'
        )

    return merged


def generate_hill_vs_alpha_table(df):
    """Generate table: Hill exponent at α=0 and versus α by family."""
    main = df[df['block'] == 'main'].copy()

    hill_agg = agg_with_bands(main, ['log_family', 'alpha'], 'hill_exponent_median')
    return hill_agg


def generate_floor_hit_table(df):
    """Generate table: Floor-hit rate by status, family, and α."""
    main = df[df['block'] == 'main'].copy()

    standalone_agg = agg_with_bands(main, ['log_family', 'alpha'], 'floor_hit_rate_standalone')
    member_agg = agg_with_bands(main, ['log_family', 'alpha'], 'floor_hit_rate_member')

    # Merge (no suffix needed, column names are unique)
    merged = standalone_agg.merge(member_agg, on=['log_family', 'alpha', 'n_reps'])
    return merged


def generate_hhi_table(df):
    """Generate table: HHI, top-10 share, and cong_capital_share versus α by family."""
    main = df[df['block'] == 'main'].copy()

    hhi_within = agg_with_bands(main, ['log_family', 'alpha'], 'hhi_within_median')
    hhi_agg = agg_with_bands(main, ['log_family', 'alpha'], 'hhi_aggregate_median')
    top10 = agg_with_bands(main, ['log_family', 'alpha'], 'top10_aggregate_median')
    ccs = agg_with_bands(main, ['log_family', 'alpha'], 'cong_capital_share_median')

    merged = hhi_within.merge(hhi_agg, on=['log_family', 'alpha', 'n_reps'],
                              suffixes=('_within', '_agg'))
    merged = merged.merge(top10, on=['log_family', 'alpha', 'n_reps'])
    merged = merged.merge(ccs, on=['log_family', 'alpha', 'n_reps'])
    merged = merged.rename(columns={
        'top10_aggregate_median_median': 'top10_median',
        'top10_aggregate_median_p25': 'top10_p25',
        'top10_aggregate_median_p75': 'top10_p75',
        'cong_capital_share_median_median': 'ccs_median',
        'cong_capital_share_median_p25': 'ccs_p25',
        'cong_capital_share_median_p75': 'ccs_p75',
    })
    return merged


def generate_equal_split_comparison(df):
    """Compare equal-split block to main normal block."""
    # Main normal block
    main_normal = df[(df['block'] == 'main') & (df['log_family'] == 'normal')].copy()
    main_k = agg_with_bands(main_normal, ['cost_type', 'alpha'], 'K_post_burnin_median')
    main_k = main_k.rename(columns={
        'K_post_burnin_median_median': 'K_main_median',
        'K_post_burnin_median_p25': 'K_main_p25',
        'K_post_burnin_median_p75': 'K_main_p75',
    })

    # Equal-split block
    equal = df[df['block'] == 'equal-split'].copy()
    equal_k = agg_with_bands(equal, ['cost_type', 'alpha'], 'K_post_burnin_median')
    equal_k = equal_k.rename(columns={
        'K_post_burnin_median_median': 'K_equal_median',
        'K_post_burnin_median_p25': 'K_equal_p25',
        'K_post_burnin_median_p75': 'K_equal_p75',
    })

    merged = main_k.merge(equal_k, on=['cost_type', 'alpha'], suffixes=('_main', '_equal'))
    return merged


def generate_lookback_comparison(df):
    """Compare lookback block to main laplace/power_law cell."""
    # Main laplace/power_law reference
    ref = df[(df['block'] == 'main') &
             (df['log_family'] == 'laplace') &
             (df['cost_type'] == 'power_law')].copy()
    ref_k = agg_with_bands(ref, ['alpha'], 'K_post_burnin_median')
    ref_k = ref_k.rename(columns={
        'K_post_burnin_median_median': 'K_ref_median',
        'K_post_burnin_median_p25': 'K_ref_p25',
        'K_post_burnin_median_p75': 'K_ref_p75',
    })
    ref_hill = agg_with_bands(ref, ['alpha'], 'hill_exponent_median')
    ref_hill = ref_hill.rename(columns={
        'hill_exponent_median_median': 'hill_ref_median',
        'hill_exponent_median_p25': 'hill_ref_p25',
        'hill_exponent_median_p75': 'hill_ref_p75',
    })
    ref_merged = ref_k.merge(ref_hill, on=['alpha', 'n_reps'])

    # Lookback block (200 and 1000)
    lookback = df[df['block'] == 'lookback'].copy()
    lookback_k = lookback.groupby(['lookback', 'alpha']).apply(
        lambda g: pd.Series({
            'K_median': g['K_post_burnin_median'].median(),
            'K_p25': np.nanpercentile(g['K_post_burnin_median'], 25),
            'K_p75': np.nanpercentile(g['K_post_burnin_median'], 75),
            'hill_median': g['hill_exponent_median'].median(),
            'hill_p25': np.nanpercentile(g['hill_exponent_median'], 25),
            'hill_p75': np.nanpercentile(g['hill_exponent_median'], 75),
            'n_reps': len(g),
        })
    ).reset_index()

    return ref_merged, lookback_k


def generate_correlation_comparison(df):
    """Compare correlation block to main laplace/power_law cell."""
    # Main laplace/power_law reference (cross_corr=0)
    ref = df[(df['block'] == 'main') &
             (df['log_family'] == 'laplace') &
             (df['cost_type'] == 'power_law')].copy()
    ref_k = agg_with_bands(ref, ['alpha'], 'K_post_burnin_median')
    ref_hill = agg_with_bands(ref, ['alpha'], 'hill_exponent_median')

    # Correlation block (cross_corr=0.3)
    corr = df[df['block'] == 'correlation'].copy()
    corr_k = agg_with_bands(corr, ['alpha'], 'K_post_burnin_median')
    corr_hill = agg_with_bands(corr, ['alpha'], 'hill_exponent_median')

    return ref_k, ref_hill, corr_k, corr_hill


def generate_endogenous_alpha_analysis(df):
    """Analyze endogenous alpha block."""
    endo = df[df['block'] == 'endogenous-alpha'].copy()

    # Summary by family/cost
    alpha_agg = endo.groupby(['log_family', 'cost_type']).apply(
        lambda g: pd.Series({
            'alpha_adopted_median': g['alpha_adopted_median'].median(),
            'alpha_adopted_mean': g['alpha_adopted_mean'].mean(),
            'alpha_adopted_std': g['alpha_adopted_std'].mean(),
            'K_median': g['K_post_burnin_median'].median(),
            'n_reps': len(g),
        })
    ).reset_index()

    return alpha_agg


def generate_runtime_stats(df):
    """Compute runtime statistics."""
    total_seconds = df['elapsed_seconds'].sum()
    mean_ms_per_step = df['ms_per_step'].mean()
    std_ms_per_step = df['ms_per_step'].std()

    return {
        'total_seconds': total_seconds,
        'total_cpu_hours': total_seconds / 3600,
        'mean_ms_per_step': mean_ms_per_step,
        'std_ms_per_step': std_ms_per_step,
        'n_scenarios': len(df),
    }


def generate_assortativity_table(df):
    """Generate table: Assortativity ratio by family/cost/alpha."""
    main = df[df['block'] == 'main'].copy()

    if 'assortativity_ratio' not in main.columns:
        return None

    assort_agg = agg_with_bands(main, ['log_family', 'cost_type', 'alpha'], 'assortativity_ratio')
    return assort_agg


def create_assortativity_figure(df, output_path='diagnostics/pilot_c_assortativity.png'):
    """Create assortativity ratio figure versus α by family."""
    main = df[(df['block'] == 'main') & (df['cost_type'] == 'power_law')].copy()

    if 'assortativity_ratio' not in main.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    families = ['normal', 'laplace', 't3']
    colors = {'normal': 'blue', 'laplace': 'orange', 't3': 'green'}

    for family in families:
        subset = main[main['log_family'] == family]
        assort_agg = agg_with_bands(subset, ['alpha'], 'assortativity_ratio')

        valid = assort_agg['assortativity_ratio_median'].notna()
        if not valid.any():
            continue

        ax.errorbar(
            assort_agg.loc[valid, 'alpha'],
            assort_agg.loc[valid, 'assortativity_ratio_median'],
            yerr=[
                assort_agg.loc[valid, 'assortativity_ratio_median'] - assort_agg.loc[valid, 'assortativity_ratio_p25'],
                assort_agg.loc[valid, 'assortativity_ratio_p75'] - assort_agg.loc[valid, 'assortativity_ratio_median']
            ],
            label=family, color=colors[family], marker='o', capsize=3
        )

    ax.axhline(1.0, color='red', linestyle='--', label='Random baseline')
    ax.set_xlabel('α (pooling fraction)')
    ax.set_ylabel('Assortativity ratio (< 1 = positive assortment)')
    ax.set_title('Member IQR SD / Random K-subset SD')
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def create_ccs_vs_alpha_figure(df, output_path='diagnostics/pilot_c_ccs_vs_alpha.png'):
    """Create conglomerate capital share versus α figure."""
    main = df[(df['block'] == 'main') & (df['log_family'] == 'laplace')].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ccs_agg = agg_with_bands(main, ['alpha'], 'cong_capital_share_median')

    ax.errorbar(
        ccs_agg['alpha'],
        ccs_agg['cong_capital_share_median_median'],
        yerr=[
            ccs_agg['cong_capital_share_median_median'] - ccs_agg['cong_capital_share_median_p25'],
            ccs_agg['cong_capital_share_median_p75'] - ccs_agg['cong_capital_share_median_median']
        ],
        color='purple', marker='o', capsize=3
    )

    ax.set_xlabel('α (pooling fraction)')
    ax.set_ylabel('Conglomerate capital share')
    ax.set_title('Conglomerate Capital Share vs α (laplace family)')
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def create_endogenous_alpha_histogram(df, output_path='diagnostics/pilot_c_endo_alpha_hist.png'):
    """Create histogram of adopted alpha values for endogenous-alpha block."""
    endo = df[df['block'] == 'endogenous-alpha'].copy()

    if len(endo) == 0 or 'alpha_adopted_median' not in endo.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    # Get adopted alpha values (median per scenario)
    alpha_values = endo['alpha_adopted_median'].dropna()

    if len(alpha_values) == 0:
        plt.close()
        return None

    ax.hist(alpha_values, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(alpha_values.median(), color='red', linestyle='--',
               label=f'Median: {alpha_values.median():.3f}')

    ax.set_xlabel('Adopted α')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Adopted α (endogenous-alpha block)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def create_hill_vs_alpha_figure(df, output_path='diagnostics/pilot_c_hill_vs_alpha.png'):
    """Create Hill exponent versus α figure by family."""
    main = df[df['block'] == 'main'].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    families = ['normal', 'laplace', 't3']
    colors = {'normal': 'blue', 'laplace': 'orange', 't3': 'green'}

    for family in families:
        subset = main[main['log_family'] == family]
        hill_agg = agg_with_bands(subset, ['alpha'], 'hill_exponent_median')

        ax.errorbar(
            hill_agg['alpha'],
            hill_agg['hill_exponent_median_median'],
            yerr=[
                hill_agg['hill_exponent_median_median'] - hill_agg['hill_exponent_median_p25'],
                hill_agg['hill_exponent_median_p75'] - hill_agg['hill_exponent_median_median']
            ],
            label=family, color=colors[family], marker='o', capsize=3
        )

    # Add theoretical exponent line at α=0
    floor_c = main['floor_c'].iloc[0] if 'floor_c' in main.columns else 0.0566
    if floor_c > 0:
        theoretical = 1.0 / (1.0 - floor_c)
        ax.axhline(theoretical, color='red', linestyle='--', label=f'1/(1-c) = {theoretical:.2f}')

    ax.set_xlabel('α (pooling fraction)')
    ax.set_ylabel('Hill exponent (median)')
    ax.set_title('Hill Exponent vs Pooling Fraction by Family')
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def create_k_vs_alpha_figure(df, output_path='diagnostics/pilot_c_k_vs_alpha.png'):
    """Create K versus α figure by family (power_law cost)."""
    main = df[(df['block'] == 'main') & (df['cost_type'] == 'power_law')].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    families = ['normal', 'laplace', 't3']
    colors = {'normal': 'blue', 'laplace': 'orange', 't3': 'green'}

    for family in families:
        subset = main[main['log_family'] == family]
        k_agg = agg_with_bands(subset, ['alpha'], 'K_post_burnin_median')

        ax.errorbar(
            k_agg['alpha'],
            k_agg['K_post_burnin_median_median'],
            yerr=[
                k_agg['K_post_burnin_median_median'] - k_agg['K_post_burnin_median_p25'],
                k_agg['K_post_burnin_median_p75'] - k_agg['K_post_burnin_median_median']
            ],
            label=family, color=colors[family], marker='o', capsize=3
        )

    ax.set_xlabel('α (pooling fraction)')
    ax.set_ylabel('K (median conglomerate size)')
    ax.set_title('Median K vs Pooling Fraction (power_law cost)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def create_floor_hit_figure(df, output_path='diagnostics/pilot_c_floor_hit.png'):
    """Create floor-hit rate figure by status and α (laplace family)."""
    main = df[(df['block'] == 'main') & (df['log_family'] == 'laplace')].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    standalone_agg = agg_with_bands(main, ['alpha'], 'floor_hit_rate_standalone')
    member_agg = agg_with_bands(main, ['alpha'], 'floor_hit_rate_member')

    ax.errorbar(
        standalone_agg['alpha'],
        standalone_agg['floor_hit_rate_standalone_median'],
        yerr=[
            standalone_agg['floor_hit_rate_standalone_median'] - standalone_agg['floor_hit_rate_standalone_p25'],
            standalone_agg['floor_hit_rate_standalone_p75'] - standalone_agg['floor_hit_rate_standalone_median']
        ],
        label='Standalone', color='red', marker='s', capsize=3
    )

    ax.errorbar(
        member_agg['alpha'],
        member_agg['floor_hit_rate_member_median'],
        yerr=[
            member_agg['floor_hit_rate_member_median'] - member_agg['floor_hit_rate_member_p25'],
            member_agg['floor_hit_rate_member_p75'] - member_agg['floor_hit_rate_member_median']
        ],
        label='Member', color='blue', marker='o', capsize=3
    )

    ax.set_xlabel('α (pooling fraction)')
    ax.set_ylabel('Floor-hit rate')
    ax.set_title('Floor-hit Rate by Status (laplace family)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def create_hhi_figure(df, output_path='diagnostics/pilot_c_hhi.png'):
    """Create HHI and top-10 share figure versus α (laplace family)."""
    main = df[(df['block'] == 'main') & (df['log_family'] == 'laplace')].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # HHI within-market
    hhi_within = agg_with_bands(main, ['alpha'], 'hhi_within_median')
    ax1.errorbar(
        hhi_within['alpha'],
        hhi_within['hhi_within_median_median'],
        yerr=[
            hhi_within['hhi_within_median_median'] - hhi_within['hhi_within_median_p25'],
            hhi_within['hhi_within_median_p75'] - hhi_within['hhi_within_median_median']
        ],
        label='Within-market', color='blue', marker='o', capsize=3
    )
    ax1.set_xlabel('α (pooling fraction)')
    ax1.set_ylabel('HHI')
    ax1.set_title('Within-market HHI vs α')
    ax1.grid(True, alpha=0.3)

    # Top-10 share
    top10 = agg_with_bands(main, ['alpha'], 'top10_aggregate_median')
    ax2.errorbar(
        top10['alpha'],
        top10['top10_aggregate_median_median'],
        yerr=[
            top10['top10_aggregate_median_median'] - top10['top10_aggregate_median_p25'],
            top10['top10_aggregate_median_p75'] - top10['top10_aggregate_median_median']
        ],
        label='Top-10 share', color='green', marker='o', capsize=3
    )
    ax2.set_xlabel('α (pooling fraction)')
    ax2.set_ylabel('Top-10 capital share')
    ax2.set_title('Aggregate Top-10 Share vs α')
    ax2.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def generate_summary_report(df, benchmarks, output_path='diagnostics/pilot_c_summary.md'):
    """Generate the full summary report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Phase C Pilot Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")

    # Runtime statistics
    runtime = generate_runtime_stats(df)
    lines.append(f"- **Scenarios**: {runtime['n_scenarios']}")
    lines.append(f"- **Total runtime**: {runtime['total_seconds']:.0f}s ({runtime['total_cpu_hours']:.2f} CPU-hours)")
    lines.append(f"- **Mean ms/step**: {runtime['mean_ms_per_step']:.2f} ± {runtime['std_ms_per_step']:.2f}")
    lines.append(f"- **Grid**: 50 × 50 (M × N)")
    lines.append("")

    # Block breakdown
    blocks = df.groupby('block').size().reset_index(name='count')
    lines.append("### Block counts")
    lines.append("")
    lines.append("| Block | Count |")
    lines.append("|-------|-------|")
    for _, row in blocks.iterrows():
        lines.append(f"| {row['block']} | {row['count']} |")
    lines.append("")

    # ========================================================================
    # Section: K and K_eff versus K*
    # ========================================================================
    lines.append("## K versus K* by Family and Cost")
    lines.append("")
    lines.append("Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `K_eff_post_burnin_median`")
    lines.append("Reference: `analytics/benchmarks.csv` column `K_star`")
    lines.append("")

    k_table = generate_k_vs_kstar_table(df, benchmarks)

    # Show for α = 0.1 (representative)
    k_sub = k_table[k_table['alpha'] == 0.1].copy()
    if len(k_sub) > 0:
        lines.append("### At α = 0.1")
        lines.append("")
        has_acc = 'acceptance_rate_median' in k_sub.columns
        if has_acc:
            lines.append("| Family | Cost | K [25,75] | K* | K_eff [25,75] | Acc. rate |")
            lines.append("|--------|------|-----------|-----|---------------|-----------|")
        else:
            lines.append("| Family | Cost | K median [25,75] | K* | K_eff median [25,75] |")
            lines.append("|--------|------|-----------------|-----|---------------------|")
        for _, row in k_sub.iterrows():
            k_band = fmt_band(row['K_post_burnin_median_median'],
                              row['K_post_burnin_median_p25'],
                              row['K_post_burnin_median_p75'])
            k_eff_band = fmt_band(row['K_eff_post_burnin_median_median'],
                                  row['K_eff_post_burnin_median_p25'],
                                  row['K_eff_post_burnin_median_p75'])
            k_star = row.get('K_star', '—')
            if pd.notna(k_star):
                k_star = int(k_star)
            if has_acc:
                acc = row.get('acceptance_rate_median', np.nan)
                acc_str = f"{acc:.3f}" if pd.notna(acc) else "—"
                lines.append(f"| {row['log_family']} | {row['cost_type']} | {k_band} | {k_star} | {k_eff_band} | {acc_str} |")
            else:
                lines.append(f"| {row['log_family']} | {row['cost_type']} | {k_band} | {k_star} | {k_eff_band} |")
        lines.append("")

    # ========================================================================
    # Section: Hill exponent
    # ========================================================================
    lines.append("## Hill Exponent")
    lines.append("")
    lines.append("Source: `pilot_c/tidy.csv` column `hill_exponent_median`")
    lines.append("")

    # Theoretical at α=0
    floor_c = df['floor_c'].iloc[0] if 'floor_c' in df.columns else 0.0566
    if floor_c > 0:
        theoretical = 1.0 / (1.0 - floor_c)
        lines.append(f"**Theoretical at α=0**: 1/(1−c) = 1/(1−{floor_c:.4f}) = {theoretical:.4f}")
        lines.append("")

    hill_table = generate_hill_vs_alpha_table(df)

    # Show α=0 by family
    hill_a0 = hill_table[hill_table['alpha'] == 0.0].copy()
    if len(hill_a0) > 0:
        lines.append("### At α = 0")
        lines.append("")
        lines.append("| Family | Hill median [25,75] |")
        lines.append("|--------|---------------------|")
        for _, row in hill_a0.iterrows():
            band = fmt_band(row['hill_exponent_median_median'],
                            row['hill_exponent_median_p25'],
                            row['hill_exponent_median_p75'])
            lines.append(f"| {row['log_family']} | {band} |")
        lines.append("")

    lines.append("### Hill exponent versus α (laplace family)")
    lines.append("")
    hill_laplace = hill_table[hill_table['log_family'] == 'laplace'].copy()
    lines.append("| α | Hill median [25,75] |")
    lines.append("|---|---------------------|")
    for _, row in hill_laplace.iterrows():
        band = fmt_band(row['hill_exponent_median_median'],
                        row['hill_exponent_median_p25'],
                        row['hill_exponent_median_p75'])
        lines.append(f"| {row['alpha']} | {band} |")
    lines.append("")

    lines.append("![Hill vs α](pilot_c_hill_vs_alpha.png)")
    lines.append("")

    # ========================================================================
    # Section: Floor-hit rate
    # ========================================================================
    lines.append("## Floor-hit Rate by Status")
    lines.append("")
    lines.append("Source: `pilot_c/tidy.csv` columns `floor_hit_rate_standalone`, `floor_hit_rate_member`")
    lines.append("")

    floor_table = generate_floor_hit_table(df)
    floor_laplace = floor_table[floor_table['log_family'] == 'laplace'].copy()

    lines.append("### Laplace family")
    lines.append("")
    lines.append("| α | Standalone [25,75] | Member [25,75] |")
    lines.append("|---|-------------------|----------------|")
    for _, row in floor_laplace.iterrows():
        standalone_band = fmt_band(row['floor_hit_rate_standalone_median'],
                                   row['floor_hit_rate_standalone_p25'],
                                   row['floor_hit_rate_standalone_p75'])
        member_band = fmt_band(row['floor_hit_rate_member_median'],
                               row['floor_hit_rate_member_p25'],
                               row['floor_hit_rate_member_p75'])
        lines.append(f"| {row['alpha']} | {standalone_band} | {member_band} |")
    lines.append("")

    lines.append("![Floor-hit rate](pilot_c_floor_hit.png)")
    lines.append("")

    # ========================================================================
    # Section: HHI and Top-10 share
    # ========================================================================
    lines.append("## HHI and Top-10 Share")
    lines.append("")
    lines.append("Source: `pilot_c/tidy.csv` columns `hhi_within_median`, `hhi_aggregate_median`, `top10_aggregate_median`")
    lines.append("")

    hhi_table = generate_hhi_table(df)
    hhi_laplace = hhi_table[hhi_table['log_family'] == 'laplace'].copy()

    lines.append("### Laplace family")
    lines.append("")
    has_ccs = 'ccs_median' in hhi_laplace.columns
    if has_ccs:
        lines.append("| α | HHI within [25,75] | HHI agg [25,75] | Top-10 [25,75] | Cong. share [25,75] |")
        lines.append("|---|-------------------|-----------------|----------------|---------------------|")
    else:
        lines.append("| α | HHI within [25,75] | HHI agg [25,75] | Top-10 [25,75] |")
        lines.append("|---|-------------------|-----------------|----------------|")
    for _, row in hhi_laplace.iterrows():
        hhi_w = fmt_band(row['hhi_within_median_median'],
                         row['hhi_within_median_p25'],
                         row['hhi_within_median_p75'])
        hhi_a = fmt_band(row['hhi_aggregate_median_median'],
                         row['hhi_aggregate_median_p25'],
                         row['hhi_aggregate_median_p75'])
        top10 = fmt_band(row.get('top10_median', np.nan),
                         row.get('top10_p25', np.nan),
                         row.get('top10_p75', np.nan))
        if has_ccs:
            ccs = fmt_band(row.get('ccs_median', np.nan),
                           row.get('ccs_p25', np.nan),
                           row.get('ccs_p75', np.nan))
            lines.append(f"| {row['alpha']} | {hhi_w} | {hhi_a} | {top10} | {ccs} |")
        else:
            lines.append(f"| {row['alpha']} | {hhi_w} | {hhi_a} | {top10} |")
    lines.append("")

    lines.append("![HHI](pilot_c_hhi.png)")
    lines.append("")
    lines.append("![Cong. Capital Share](pilot_c_ccs_vs_alpha.png)")
    lines.append("")

    # ========================================================================
    # Section: Equal-split check
    # ========================================================================
    lines.append("## Equal-split Check")
    lines.append("")
    lines.append("Comparison of equal-split block to main normal block.")
    lines.append("Source: `pilot_c/tidy.csv` column `K_post_burnin_median`, filtered by `block` and `sharing_rule`")
    lines.append("")

    equal_table = generate_equal_split_comparison(df)
    # Show for α = 0.1
    equal_sub = equal_table[equal_table['alpha'] == 0.1].copy()
    if len(equal_sub) > 0:
        lines.append("### At α = 0.1")
        lines.append("")
        lines.append("| Cost | K (proportional) [25,75] | K (equal) [25,75] |")
        lines.append("|------|--------------------------|-------------------|")
        for _, row in equal_sub.iterrows():
            k_main = fmt_band(row['K_main_median'], row['K_main_p25'], row['K_main_p75'])
            k_equal = fmt_band(row['K_equal_median'], row['K_equal_p25'], row['K_equal_p75'])
            lines.append(f"| {row['cost_type']} | {k_main} | {k_equal} |")
        lines.append("")

    # ========================================================================
    # Section: Lookback comparison
    # ========================================================================
    lines.append("## Lookback Sensitivity")
    lines.append("")
    lines.append("Comparison of lookback block (200, 1000) to main laplace/power_law cell (50).")
    lines.append("Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `hill_exponent_median`, filtered by `block` and `lookback`")
    lines.append("")

    ref_look, lookback_k = generate_lookback_comparison(df)

    # Show for α = 0.1
    ref_a01 = ref_look[ref_look['alpha'] == 0.1]
    look_a01 = lookback_k[lookback_k['alpha'] == 0.1]

    if len(ref_a01) > 0:
        lines.append("### At α = 0.1")
        lines.append("")
        lines.append("| Lookback | K [25,75] | Hill [25,75] |")
        lines.append("|----------|-----------|--------------|")
        for _, row in ref_a01.iterrows():
            k_band = fmt_band(row['K_ref_median'], row['K_ref_p25'], row['K_ref_p75'])
            h_band = fmt_band(row['hill_ref_median'], row['hill_ref_p25'], row['hill_ref_p75'])
            lines.append(f"| 50 (ref) | {k_band} | {h_band} |")
        for _, row in look_a01.iterrows():
            k_band = fmt_band(row['K_median'], row['K_p25'], row['K_p75'])
            h_band = fmt_band(row['hill_median'], row['hill_p25'], row['hill_p75'])
            lines.append(f"| {int(row['lookback'])} | {k_band} | {h_band} |")
        lines.append("")

    # ========================================================================
    # Section: Correlation comparison
    # ========================================================================
    lines.append("## Correlation Sensitivity")
    lines.append("")
    lines.append("Comparison of correlation block (cross_corr=0.3) to main laplace/power_law cell (cross_corr=0).")
    lines.append("Source: `pilot_c/tidy.csv` columns `K_post_burnin_median`, `hill_exponent_median`, filtered by `block` and `cross_corr`")
    lines.append("")

    ref_k, ref_h, corr_k, corr_h = generate_correlation_comparison(df)

    # Show for α = 0.1
    ref_k_a01 = ref_k[ref_k['alpha'] == 0.1]
    ref_h_a01 = ref_h[ref_h['alpha'] == 0.1]
    corr_k_a01 = corr_k[corr_k['alpha'] == 0.1]
    corr_h_a01 = corr_h[corr_h['alpha'] == 0.1]

    if len(ref_k_a01) > 0:
        lines.append("### At α = 0.1")
        lines.append("")
        lines.append("| Cross-corr | K [25,75] | Hill [25,75] |")
        lines.append("|------------|-----------|--------------|")
        for _, row in ref_k_a01.iterrows():
            k_band = fmt_band(row['K_post_burnin_median_median'],
                              row['K_post_burnin_median_p25'],
                              row['K_post_burnin_median_p75'])
            lines.append(f"| 0.0 (ref) | {k_band} | — |")
        for _, row in ref_h_a01.iterrows():
            h_band = fmt_band(row['hill_exponent_median_median'],
                              row['hill_exponent_median_p25'],
                              row['hill_exponent_median_p75'])
            # Update previous row
            lines[-1] = lines[-1].replace("| — |", f"| {h_band} |")
        for i, (_, k_row) in enumerate(corr_k_a01.iterrows()):
            k_band = fmt_band(k_row['K_post_burnin_median_median'],
                              k_row['K_post_burnin_median_p25'],
                              k_row['K_post_burnin_median_p75'])
            h_row = corr_h_a01.iloc[i] if i < len(corr_h_a01) else None
            if h_row is not None:
                h_band = fmt_band(h_row['hill_exponent_median_median'],
                                  h_row['hill_exponent_median_p25'],
                                  h_row['hill_exponent_median_p75'])
            else:
                h_band = "—"
            lines.append(f"| 0.3 | {k_band} | {h_band} |")
        lines.append("")

    # ========================================================================
    # Section: Endogenous α
    # ========================================================================
    lines.append("## Endogenous α")
    lines.append("")
    lines.append("Analysis of endogenous-alpha block.")
    lines.append("Source: `pilot_c/tidy.csv` columns `alpha_adopted_median`, `alpha_adopted_mean`, `alpha_adopted_std`")
    lines.append("")

    endo_table = generate_endogenous_alpha_analysis(df)

    if len(endo_table) > 0:
        lines.append("### Adopted α by family and cost")
        lines.append("")
        lines.append("| Family | Cost | α adopted (median) | α adopted (mean) | K |")
        lines.append("|--------|------|--------------------|------------------|---|")
        for _, row in endo_table.iterrows():
            alpha_med = row['alpha_adopted_median']
            alpha_mean = row['alpha_adopted_mean']
            K_med = row['K_median']
            alpha_med_str = f"{alpha_med:.3f}" if not pd.isna(alpha_med) else "—"
            alpha_mean_str = f"{alpha_mean:.3f}" if not pd.isna(alpha_mean) else "—"
            K_str = f"{K_med:.1f}" if not pd.isna(K_med) else "—"
            lines.append(f"| {row['log_family']} | {row['cost_type']} | {alpha_med_str} | {alpha_mean_str} | {K_str} |")
        lines.append("")

    lines.append("![Endogenous α histogram](pilot_c_endo_alpha_hist.png)")
    lines.append("")

    # ========================================================================
    # Section: Assortativity
    # ========================================================================
    lines.append("## Assortativity")
    lines.append("")
    lines.append("SD of member IQR / SD of random K-subset. Ratio < 1 indicates positive assortment")
    lines.append("(conglomerate members have more similar growth volatilities than random groups).")
    lines.append("")
    lines.append("Source: `pilot_c/tidy.csv` column `assortativity_ratio`")
    lines.append("")

    assort_table = generate_assortativity_table(df)
    if assort_table is not None and len(assort_table) > 0:
        # Show for α = 0.1
        assort_sub = assort_table[assort_table['alpha'] == 0.1].copy()
        if len(assort_sub) > 0:
            lines.append("### At α = 0.1 (power_law cost)")
            lines.append("")
            lines.append("| Family | Assort. ratio [25,75] |")
            lines.append("|--------|----------------------|")
            for _, row in assort_sub[assort_sub['cost_type'] == 'power_law'].iterrows():
                band = fmt_band(row['assortativity_ratio_median'],
                                row['assortativity_ratio_p25'],
                                row['assortativity_ratio_p75'])
                lines.append(f"| {row['log_family']} | {band} |")
            lines.append("")

        lines.append("![Assortativity](pilot_c_assortativity.png)")
        lines.append("")
    else:
        lines.append("*Assortativity data not available.*")
        lines.append("")

    # ========================================================================
    # Section: Runtime
    # ========================================================================
    lines.append("## Runtime Statistics")
    lines.append("")
    lines.append(f"- **Total runtime**: {runtime['total_seconds']:.0f}s ({runtime['total_cpu_hours']:.2f} CPU-hours)")
    lines.append(f"- **Mean ms/step**: {runtime['mean_ms_per_step']:.2f} ± {runtime['std_ms_per_step']:.2f}")
    lines.append("")

    # ========================================================================
    # Footer
    # ========================================================================
    lines.append("## Figures")
    lines.append("")
    lines.append("- `pilot_c_hill_vs_alpha.png`: Hill exponent versus α by family")
    lines.append("- `pilot_c_k_vs_alpha.png`: K versus α by family (power_law cost)")
    lines.append("- `pilot_c_floor_hit.png`: Floor-hit rate by status (laplace family)")
    lines.append("- `pilot_c_hhi.png`: HHI and top-10 share versus α (laplace family)")
    lines.append("- `pilot_c_ccs_vs_alpha.png`: Conglomerate capital share versus α")
    lines.append("- `pilot_c_assortativity.png`: Assortativity ratio versus α")
    lines.append("- `pilot_c_endo_alpha_hist.png`: Histogram of adopted α (endogenous block)")
    lines.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


def main():
    """Run Phase C pilot analysis."""
    print("Phase C Pilot Analysis")
    print("=" * 50)

    # Load data
    print("Loading tidy.csv...")
    df = load_tidy_csv()
    print(f"  {len(df)} scenarios")

    print("Loading benchmarks.csv...")
    try:
        benchmarks = load_benchmarks()
        print(f"  {len(benchmarks)} benchmark rows")
    except FileNotFoundError:
        print("  WARNING: benchmarks.csv not found, skipping K* comparisons")
        benchmarks = pd.DataFrame()

    # Generate figures
    print("Generating figures...")
    create_hill_vs_alpha_figure(df)
    create_k_vs_alpha_figure(df)
    create_floor_hit_figure(df)
    create_hhi_figure(df)
    create_ccs_vs_alpha_figure(df)
    create_assortativity_figure(df)
    create_endogenous_alpha_histogram(df)

    # Generate report
    print("Generating summary report...")
    output_path = generate_summary_report(df, benchmarks)
    print(f"Wrote {output_path}")


if __name__ == '__main__':
    main()
