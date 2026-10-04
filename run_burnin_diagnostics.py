#!/usr/bin/env python3
"""
Burn-in diagnostics for each cost type baseline.

Runs simulations with sufficient timesteps to analyze convergence,
then produces rolling-mean plots and a summary markdown report.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the model
from collaborative_growth import model


def run_simulation(cost_type, seed=42, steps=5000):
    """
    Run a single simulation for burn-in analysis.

    Parameters:
    -----------
    cost_type : str
        One of 'power_law', 'exponential', 'linear', 'quadratic'
    seed : int
        Random seed
    steps : int
        Number of timesteps

    Returns:
    --------
    dict with time series for analysis
    """
    M, N = 100, 100

    # Cost function parameters (baseline values)
    cost_params = {
        'power_law': (0.0001, 1.2, 0.001),
        'exponential': (0.005, 0.2, 0.001),
        'linear': (0.005, 0.005, 0.001),
        'quadratic': (0.005, 0.005, 0.001),
    }
    c0, c1, c2 = cost_params[cost_type]

    params = [M, N, steps, 0.1, M * N, 0.05, 4, 0.85, False, 50,
              cost_type, c0, c1, c2]

    print(f"  Running {cost_type} simulation (M={M}, N={N}, T={steps})...")
    result = model(params, seed=seed, market_corr='identity')

    # Unpack results (13 elements with online rank stats)
    (mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
     gini_coefficient, avg_ranks, mergers_per_period, proposals_per_period,
     exits_per_period, rank_range, rank_std, hyperparameters) = result

    # gini_coefficient has shape (markets, steps) - average over markets
    gini_avg = np.mean(gini_coefficient, axis=0)

    # num_cong is a list - convert to array
    num_cong_arr = np.array(num_cong)

    return {
        'cost_type': cost_type,
        'gini': gini_avg,
        'mean_members': mean_members,
        'num_cong': num_cong_arr,
        'mergers': mergers_per_period,
        'exits': exits_per_period,
        'steps': steps,
    }


def compute_rolling_mean(series, window=100):
    """Compute rolling mean with given window size."""
    return np.convolve(series, np.ones(window)/window, mode='valid')


def analyze_convergence(series, window=100, threshold=0.01):
    """
    Analyze convergence using rolling mean stability.

    The rolling mean is considered stable when the coefficient of variation
    of the rolling mean in the late portion is below threshold.

    Returns:
    --------
    dict with analysis results
    """
    rolling = compute_rolling_mean(series, window)

    # Compare early vs late rolling mean
    n = len(rolling)
    early = rolling[:n//4]
    late = rolling[-n//4:]

    mean_early = np.mean(early)
    mean_late = np.mean(late)
    std_late = np.std(late)

    # Coefficient of variation in late portion
    cv_late = std_late / max(abs(mean_late), 1e-10)

    # Relative drift between early and late
    drift = abs(mean_late - mean_early) / max(abs(mean_late), 1e-10)

    # Determine recommended burn-in
    # Find first point where rolling mean stabilizes (within 5% of final mean)
    final_mean = np.mean(rolling[-n//10:])
    tolerance = 0.05 * max(abs(final_mean), 0.01)

    burnin_idx = 0
    for i in range(len(rolling)):
        if abs(rolling[i] - final_mean) <= tolerance:
            burnin_idx = i
            break

    return {
        'rolling_mean': rolling,
        'mean_early': mean_early,
        'mean_late': mean_late,
        'cv_late': cv_late,
        'drift': drift,
        'converged': cv_late < threshold and drift < threshold,
        'recommended_burnin': burnin_idx + window,  # Account for window offset
    }


def create_burnin_figure(results, output_path):
    """
    Create a figure showing rolling mean convergence.

    Parameters:
    -----------
    results : dict
        Simulation results
    output_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('gini', 'Gini Coefficient'),
        ('mean_members', 'Mean Conglomerate Size'),
        ('num_cong', 'Number of Conglomerates'),
        ('mergers', 'Mergers per Period'),
    ]

    for ax, (key, title) in zip(axes.flat, metrics):
        series = results[key]

        # Plot raw series
        ax.plot(series, alpha=0.3, color='gray', label='Raw', linewidth=0.5)

        # Plot rolling mean
        rolling = compute_rolling_mean(series, window=100)
        t_rolling = np.arange(len(rolling)) + 100 - 1  # Center of window
        ax.plot(t_rolling, rolling, color='blue', label='Rolling Mean (w=100)', linewidth=1.5)

        # Mark recommended burn-in
        analysis = analyze_convergence(series)
        burnin = analysis['recommended_burnin']
        ax.axvline(x=burnin, color='red', linestyle='--', label=f'Burn-in ({burnin})')

        ax.set_xlabel('Time Step')
        ax.set_ylabel(title)
        ax.set_title(f'{title} - {results["cost_type"]}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Burn-in Diagnostics: {results["cost_type"]} Cost Function', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figure saved: {output_path}")


def generate_markdown_report(all_results, output_path, figures_dir):
    """
    Generate markdown report with burn-in analysis.

    Parameters:
    -----------
    all_results : dict
        Results for all cost types
    output_path : str
        Path to save markdown file
    figures_dir : str
        Path to figures directory (relative to markdown file)
    """
    lines = [
        "# Burn-in Diagnostics",
        "",
        "Analysis of simulation convergence for each cost function type.",
        "",
        "## Rolling Mean Criterion",
        "",
        "The rolling mean with window size 100 is used to assess stationarity.",
        "Convergence is determined by:",
        "- Coefficient of variation in late portion < 1%",
        "- Relative drift between early and late means < 1%",
        "",
        "## Summary Table",
        "",
        "| Cost Type | Metric | Early Mean | Late Mean | CV (Late) | Drift | Converged | Burn-in |",
        "|-----------|--------|------------|-----------|-----------|-------|-----------|---------|",
    ]

    for cost_type, results in all_results.items():
        for metric in ['gini', 'mean_members', 'num_cong', 'mergers']:
            series = results[metric]
            analysis = analyze_convergence(series)

            metric_name = {
                'gini': 'Gini',
                'mean_members': 'Mean Size',
                'num_cong': 'Num Cong',
                'mergers': 'Mergers',
            }[metric]

            converged = '✓' if analysis['converged'] else '✗'

            lines.append(
                f"| {cost_type} | {metric_name} | "
                f"{analysis['mean_early']:.4f} | {analysis['mean_late']:.4f} | "
                f"{analysis['cv_late']:.4f} | {analysis['drift']:.4f} | "
                f"{converged} | {analysis['recommended_burnin']} |"
            )

    lines.extend([
        "",
        "## Figures by Cost Type",
        "",
    ])

    for cost_type in all_results.keys():
        fig_path = f"{figures_dir}/burnin_{cost_type}.png"
        lines.extend([
            f"### {cost_type.replace('_', ' ').title()}",
            "",
            f"![{cost_type} burn-in]({fig_path})",
            "",
        ])

    lines.extend([
        "## Conclusion",
        "",
        "Based on the rolling mean analysis:",
        "",
    ])

    # Check overall convergence
    all_converged = True
    max_burnin = 0
    for cost_type, results in all_results.items():
        for metric in ['gini', 'mean_members']:
            analysis = analyze_convergence(results[metric])
            if not analysis['converged']:
                all_converged = False
            max_burnin = max(max_burnin, analysis['recommended_burnin'])

    if all_converged:
        lines.append(f"- All simulations show convergence of key metrics (Gini, Mean Size)")
        lines.append(f"- Recommended burn-in period: **{max_burnin} steps** (maximum across all cost types)")
        lines.append(f"- This represents {100*max_burnin/5000:.1f}% of the simulation length")
    else:
        lines.append("- Some metrics show incomplete convergence")
        lines.append("- Consider increasing simulation length or adjusting parameters")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report saved: {output_path}")


def main():
    """Run burn-in diagnostics for all cost types."""
    # Output directories
    output_dir = Path('diagnostics')
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    cost_types = ['power_law', 'exponential', 'linear', 'quadratic']
    all_results = {}

    print("Running burn-in diagnostics...")
    print("=" * 50)

    for cost_type in cost_types:
        print(f"\n{cost_type}:")
        results = run_simulation(cost_type, seed=42, steps=5000)
        all_results[cost_type] = results

        # Create figure
        fig_path = figures_dir / f'burnin_{cost_type}.png'
        create_burnin_figure(results, fig_path)

    # Generate markdown report
    print("\nGenerating report...")
    report_path = output_dir / 'burn_in.md'
    generate_markdown_report(all_results, report_path, 'figures')

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == '__main__':
    main()
