#!/usr/bin/env python3
"""
Pilot study analysis (§7).

Produces diagnostics/pilot_summary.md with:
- Per (sharing_rule, sigma_range, rho): table over cost_type × α
- Metrics: acceptance rate, mean size, num conglomerates, exits/period,
           share quantiles 50/99/100, median rank range

Figures:
1. Acceptance rate and mean size vs α (lines by sigma_range, panels by sharing rule)
2. Share quantiles (50/99/100) vs α (same layout)
3. α=0 before/after bars for 100th-percentile share by cost type

Usage:
    python analyze_pilot.py
    python analyze_pilot.py --results-dir results/pilot
"""
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(results_dir='results/pilot'):
    """Load all pilot result files."""
    results_path = Path(results_dir)
    results = []

    for pkl_file in results_path.glob('*.pkl'):
        try:
            with open(pkl_file, 'rb') as f:
                result = pickle.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {pkl_file}: {e}")

    print(f"Loaded {len(results)} result files")
    return results


def results_to_dataframe(results):
    """Convert results list to DataFrame."""
    rows = []
    for r in results:
        # Handle sigma_range tuple
        sigma_range = r.get('sigma_range', (0.0, 0.0))
        if isinstance(sigma_range, tuple):
            sigma_lo, sigma_hi = sigma_range
        else:
            sigma_lo, sigma_hi = 0.0, 0.0

        rows.append({
            'scenario_name': r['scenario_name'],
            'alpha': r['alpha'],
            'rep': r['rep'],
            'sharing_rule': r['sharing_rule'],
            'sigma_lo': sigma_lo,
            'sigma_hi': sigma_hi,
            'sigma_range': f"{sigma_lo:.2f}-{sigma_hi:.2f}",
            'rho': r['rho'],
            'cost_type': r['cost_type'],
            'acceptance_rate': r['acceptance_rate'],
            'mean_size': r['mean_size'],
            'num_cong': r['num_cong'],
            'exits_per_period': r['exits_per_period'],
            'share_q50': r['share_q50'],
            'share_q99': r['share_q99'],
            'share_q100': r['share_q100'],
            'rank_range_median': r['rank_range_median'],
            'gini': r['gini'],
        })

    return pd.DataFrame(rows)


def aggregate_by_treatment(df):
    """Aggregate results by treatment combination (mean over replications)."""
    group_cols = ['scenario_name', 'sharing_rule', 'sigma_range', 'rho', 'cost_type', 'alpha']
    agg_cols = ['acceptance_rate', 'mean_size', 'num_cong', 'exits_per_period',
                'share_q50', 'share_q99', 'share_q100', 'rank_range_median', 'gini']

    agg = df.groupby(group_cols)[agg_cols].agg(['mean', 'std']).reset_index()

    # Flatten column names
    agg.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg.columns]

    return agg


def generate_treatment_table(df_agg, sharing_rule, sigma_range, rho):
    """Generate markdown table for a specific treatment combination."""
    # Filter data
    mask = (
        (df_agg['sharing_rule'] == sharing_rule) &
        (df_agg['sigma_range'] == sigma_range) &
        (df_agg['rho'] == rho)
    )
    subset = df_agg[mask].copy()

    if len(subset) == 0:
        return None

    # Pivot: rows = cost_type, columns = alpha
    alpha_values = sorted(subset['alpha'].unique())
    cost_types = sorted(subset['cost_type'].unique())

    lines = []
    lines.append(f"### {sharing_rule.title()} Sharing, σ ∈ {sigma_range}, ρ = {rho}")
    lines.append("")

    # Table header
    header = "| Metric | Cost Type | " + " | ".join([f"α={a:.2f}" for a in alpha_values]) + " |"
    sep = "|--------|-----------|" + "|".join(["-------" for _ in alpha_values]) + "|"
    lines.append(header)
    lines.append(sep)

    metrics = [
        ('acceptance_rate_mean', 'Accept Rate'),
        ('mean_size_mean', 'Mean Size'),
        ('num_cong_mean', 'Num Cong'),
        ('exits_per_period_mean', 'Exits/Period'),
        ('share_q50_mean', 'Share Q50'),
        ('share_q99_mean', 'Share Q99'),
        ('share_q100_mean', 'Share Q100'),
        ('rank_range_median_mean', 'Rank Range'),
    ]

    for metric_col, metric_name in metrics:
        for cost_type in cost_types:
            values = []
            for alpha in alpha_values:
                row = subset[(subset['cost_type'] == cost_type) & (subset['alpha'] == alpha)]
                if len(row) > 0:
                    val = row[metric_col].values[0]
                    if 'rate' in metric_col or metric_col.startswith('share'):
                        values.append(f"{val:.4f}")
                    elif 'size' in metric_col or 'cong' in metric_col or 'rank' in metric_col:
                        values.append(f"{val:.2f}")
                    else:
                        values.append(f"{val:.3f}")
                else:
                    values.append("-")
            lines.append(f"| {metric_name} | {cost_type} | " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_summary_report(df, df_agg, output_path='diagnostics/pilot_summary.md'):
    """Generate full markdown summary report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Pilot Study Summary",
        "",
        "Analysis of 48 factorial treatment scenarios + reference scenarios.",
        "",
        "## Design",
        "",
        "- **Fixed**: M=N=50, T=5000, burn_in=4000, lookback=50, p=0.05",
        "- **Growth**: lognormal, μ ∈ [0.01, 0.1]",
        "- **Replications**: 5 with common random numbers",
        "- **α grid**: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]",
        "",
        "### Factorial Treatments",
        "",
        "| Factor | Levels |",
        "|--------|--------|",
        "| sharing_rule | equal, proportional |",
        "| sigma_range | [0.05, 0.10], [0.10, 0.20], [0.20, 0.40] |",
        "| rho | 0, 0.3 |",
        "| cost_type | linear, quadratic, exponential, power_law |",
        "",
        f"Total: {len(df['scenario_name'].unique())} scenarios × 9 α × 5 reps = {len(df)} runs",
        "",
        "---",
        "",
        "## Results by Treatment Combination",
        "",
    ]

    # Get unique treatment combinations
    sharing_rules = sorted(df_agg['sharing_rule'].unique())
    sigma_ranges = sorted(df_agg['sigma_range'].unique())
    rhos = sorted(df_agg['rho'].unique())

    for sharing_rule in sharing_rules:
        for sigma_range in sigma_ranges:
            for rho in rhos:
                table = generate_treatment_table(df_agg, sharing_rule, sigma_range, rho)
                if table:
                    lines.append(table)

    # Write report
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Report saved: {output_path}")


def create_figure_1(df_agg, output_path='diagnostics/figures/pilot_fig1_acceptance_size.png'):
    """
    Figure 1: Acceptance rate and mean size vs α.
    Lines by sigma_range, panels by sharing rule.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sharing_rules = ['equal', 'proportional']
    metrics = [('acceptance_rate_mean', 'Acceptance Rate'), ('mean_size_mean', 'Mean Conglomerate Size')]
    sigma_ranges = sorted(df_agg['sigma_range'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sigma_ranges)))

    for col, sharing_rule in enumerate(sharing_rules):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]

            for sigma_range, color in zip(sigma_ranges, colors):
                # Average over rho and cost_type
                mask = (df_agg['sharing_rule'] == sharing_rule) & (df_agg['sigma_range'] == sigma_range)
                subset = df_agg[mask].groupby('alpha')[metric].mean().reset_index()

                ax.plot(subset['alpha'], subset[metric], 'o-', color=color,
                       label=f'σ ∈ {sigma_range}', linewidth=2, markersize=6)

            ax.set_xlabel('α (sharing parameter)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{sharing_rule.title()} Sharing')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Pilot Study: Acceptance Rate and Mean Size vs α', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure 1 saved: {output_path}")


def create_figure_2(df_agg, output_path='diagnostics/figures/pilot_fig2_share_quantiles.png'):
    """
    Figure 2: 50th/99th/100th share quantiles vs α.
    Lines by sigma_range, panels by sharing rule.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    sharing_rules = ['equal', 'proportional']
    quantiles = [('share_q50_mean', '50th Percentile'),
                 ('share_q99_mean', '99th Percentile'),
                 ('share_q100_mean', '100th Percentile (Max)')]
    sigma_ranges = sorted(df_agg['sigma_range'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sigma_ranges)))

    for col, sharing_rule in enumerate(sharing_rules):
        for row, (metric, ylabel) in enumerate(quantiles):
            ax = axes[row, col]

            for sigma_range, color in zip(sigma_ranges, colors):
                # Average over rho and cost_type
                mask = (df_agg['sharing_rule'] == sharing_rule) & (df_agg['sigma_range'] == sigma_range)
                subset = df_agg[mask].groupby('alpha')[metric].mean().reset_index()

                ax.plot(subset['alpha'], subset[metric], 'o-', color=color,
                       label=f'σ ∈ {sigma_range}', linewidth=2, markersize=6)

            ax.set_xlabel('α (sharing parameter)')
            ax.set_ylabel(f'Market Share ({ylabel})')
            ax.set_title(f'{sharing_rule.title()} Sharing - {ylabel}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Pilot Study: Share Quantiles vs α', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure 2 saved: {output_path}")


def create_figure_3(df_agg, output_path='diagnostics/figures/pilot_fig3_alpha0_comparison.png'):
    """
    Figure 3: α=0 before/after bars for 100th-percentile share by cost type.
    Compares pilot results at α=0 across cost types.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Filter to α=0
    alpha0 = df_agg[df_agg['alpha'] == 0.0].copy()

    if len(alpha0) == 0:
        print("Warning: No α=0 data found for Figure 3")
        return

    # Group by cost_type, averaging over other treatments
    cost_types = ['linear', 'quadratic', 'exponential', 'power_law']
    x = np.arange(len(cost_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get average share_q100 by cost_type for equal vs proportional
    equal_vals = []
    prop_vals = []

    for cost_type in cost_types:
        mask_equal = (alpha0['cost_type'] == cost_type) & (alpha0['sharing_rule'] == 'equal')
        mask_prop = (alpha0['cost_type'] == cost_type) & (alpha0['sharing_rule'] == 'proportional')

        equal_vals.append(alpha0[mask_equal]['share_q100_mean'].mean() if mask_equal.any() else 0)
        prop_vals.append(alpha0[mask_prop]['share_q100_mean'].mean() if mask_prop.any() else 0)

    bars1 = ax.bar(x - width/2, equal_vals, width, label='Equal Sharing', color='steelblue')
    bars2 = ax.bar(x + width/2, prop_vals, width, label='Proportional Sharing', color='coral')

    ax.set_xlabel('Cost Function Type')
    ax.set_ylabel('Max Market Share (100th Percentile)')
    ax.set_title('α = 0: Maximum Market Share by Cost Type and Sharing Rule')
    ax.set_xticks(x)
    ax.set_xticklabels(cost_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure 3 saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze pilot study results')
    parser.add_argument('--results-dir', type=str, default='results/pilot',
                       help='Directory containing result pickle files')
    parser.add_argument('--output-dir', type=str, default='diagnostics',
                       help='Output directory for report and figures')

    args = parser.parse_args()

    # Load results
    results = load_results(args.results_dir)

    if len(results) == 0:
        print("No results found. Run pilot study first.")
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)
    print(f"DataFrame shape: {df.shape}")
    print(f"Scenarios: {df['scenario_name'].nunique()}")
    print(f"α values: {sorted(df['alpha'].unique())}")

    # Aggregate
    df_agg = aggregate_by_treatment(df)

    # Generate report
    report_path = f"{args.output_dir}/pilot_summary.md"
    generate_summary_report(df, df_agg, report_path)

    # Create figures
    figures_dir = f"{args.output_dir}/figures"
    create_figure_1(df_agg, f"{figures_dir}/pilot_fig1_acceptance_size.png")
    create_figure_2(df_agg, f"{figures_dir}/pilot_fig2_share_quantiles.png")
    create_figure_3(df_agg, f"{figures_dir}/pilot_fig3_alpha0_comparison.png")

    print("\nAnalysis complete!")
    print(f"  Report: {report_path}")
    print(f"  Figures: {figures_dir}/pilot_fig*.png")


if __name__ == '__main__':
    main()
