#!/usr/bin/env python3
"""
C13: Calibrate floor coefficient c at N=50.

Runs a grid search over floor_c values to find c_star where the
within-market Hill exponent equals 1.06 (Axtell 2001 target).

Configuration:
- α=0 (no pooling)
- market_size_fixed=True
- log_family='laplace'
- IQR ∈ [0.1, 0.3] per market
- M=20, N=50, T=8000, burn_in=3000
- 5 reps per c value
- metric_every=100

Also runs N=200 at selected c values to show finite-N correction shrinking.
"""
import numpy as np
import sys
import os
import csv
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collaborative_growth import model

# Configuration
M = 20
T = 8000
BURN_IN = 3000
REPS = 5
METRIC_EVERY = 100
SEED_BASE = 42

# Grid of c values for N=50
C_GRID_N50 = [0.03, 0.05, 0.0566, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# Subset of c values for N=200
C_GRID_N200 = [0.0566, 0.15, 0.3]

# Target Hill exponent (Axtell 2001)
TARGET_HILL = 1.06


def run_calibration_scenario(N, c, rep, seed):
    """
    Run a single calibration scenario.

    Returns dict with metrics.
    """
    total_firms = M * N

    # Build params: 10 elements (defaults to power_law, but doesn't matter for α=0)
    # [M, N, T, share, total_firms, merge_thresh, min_size, exit_prob, proportional, lookback]
    params = [M, N, T, 0.0, total_firms, 0.001, 1, 0.001, False, 50]

    results = model(
        params, seed=seed,
        growth_process='log_family', log_family='laplace',
        sigma_range=(0.1, 0.3), g=0.0,
        floor_c=c,
        market_size_fixed=True,
        metric_every=METRIC_EVERY,
        burn_in=BURN_IN
    )

    hyperparameters = results[-1]
    summary = hyperparameters['summary']

    # Hill exponent: pooled within-market shares, median over post-burn-in
    hill_median = summary['hill_exponent_median']

    # HHI within markets: per-market median over post-burn-in
    hhi_within_median = np.nanmedian(summary['hhi_within_median'])

    # Top10 aggregate
    top10_median = summary['top10_aggregate_median']

    # Floor hit rate (fraction of firms at floor per period)
    # Use total floor hits / (total firms * post-burn-in periods)
    floor_hits_by_status = hyperparameters['floor_hits_by_status']
    post_burn_steps = T - BURN_IN
    total_hits = floor_hits_by_status[BURN_IN:].sum()
    floor_frac = total_hits / (total_firms * post_burn_steps) if post_burn_steps > 0 else 0.0

    return {
        'N': N,
        'c': c,
        'rep': rep,
        'seed': seed,
        'hill_median': hill_median,
        'hhi_within_median': hhi_within_median,
        'top10_median': top10_median,
        'floor_frac': floor_frac,
    }


def aggregate_results(results_list):
    """
    Aggregate results across reps for each (N, c) combination.

    Returns list of dicts with median and 25-75% bands.
    """
    from collections import defaultdict

    # Group by (N, c)
    groups = defaultdict(list)
    for r in results_list:
        key = (r['N'], r['c'])
        groups[key].append(r)

    aggregated = []
    for (N, c), reps in sorted(groups.items()):
        hills = [r['hill_median'] for r in reps]
        hhis = [r['hhi_within_median'] for r in reps]
        top10s = [r['top10_median'] for r in reps]
        floor_fracs = [r['floor_frac'] for r in reps]

        aggregated.append({
            'N': N,
            'c': c,
            'theoretical_hill': 1.0 / (1.0 - c),
            'hill_median': np.median(hills),
            'hill_p25': np.percentile(hills, 25),
            'hill_p75': np.percentile(hills, 75),
            'hhi_median': np.median(hhis),
            'hhi_p25': np.percentile(hhis, 25),
            'hhi_p75': np.percentile(hhis, 75),
            'top10_median': np.median(top10s),
            'top10_p25': np.percentile(top10s, 25),
            'top10_p75': np.percentile(top10s, 75),
            'floor_frac_median': np.median(floor_fracs),
            'floor_frac_p25': np.percentile(floor_fracs, 25),
            'floor_frac_p75': np.percentile(floor_fracs, 75),
        })

    return aggregated


def find_c_star(aggregated, N=50):
    """
    Find c_star by linear interpolation where Hill = TARGET_HILL.

    Returns (c_star, hill_at_c_star) or (None, max_hill) if target not reached.
    """
    # Filter to N=50 results
    n50_results = [r for r in aggregated if r['N'] == N]
    n50_results.sort(key=lambda x: x['c'])

    # Check if we ever reach the target
    max_hill = max(r['hill_median'] for r in n50_results)
    if max_hill < TARGET_HILL:
        return None, max_hill

    # Linear interpolation
    for i in range(len(n50_results) - 1):
        c1, h1 = n50_results[i]['c'], n50_results[i]['hill_median']
        c2, h2 = n50_results[i + 1]['c'], n50_results[i + 1]['hill_median']

        # Check if target is between these two points
        if (h1 <= TARGET_HILL <= h2) or (h2 <= TARGET_HILL <= h1):
            # Linear interpolation
            if abs(h2 - h1) < 1e-10:
                c_star = (c1 + c2) / 2
            else:
                c_star = c1 + (TARGET_HILL - h1) * (c2 - c1) / (h2 - h1)
            # Compute hill at c_star by interpolation
            hill_at_c_star = TARGET_HILL
            return c_star, hill_at_c_star

    # Target may be at an endpoint
    for r in n50_results:
        if abs(r['hill_median'] - TARGET_HILL) < 0.01:
            return r['c'], r['hill_median']

    return None, max_hill


def write_csv(aggregated, output_path):
    """Write results to CSV."""
    fieldnames = [
        'N', 'c', 'theoretical_hill',
        'hill_median', 'hill_p25', 'hill_p75',
        'hhi_median', 'hhi_p25', 'hhi_p75',
        'top10_median', 'top10_p25', 'top10_p75',
        'floor_frac_median', 'floor_frac_p25', 'floor_frac_p75',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated)

    print(f"Wrote {output_path}")


def write_markdown(aggregated, c_star, hill_at_c_star, output_path):
    """Write markdown summary with table and figure description."""
    n50_results = [r for r in aggregated if r['N'] == 50]
    n200_results = [r for r in aggregated if r['N'] == 200]

    lines = [
        "# Floor Calibration Results",
        "",
        "## Configuration",
        "",
        f"- M = {M}, T = {T}, burn_in = {BURN_IN}",
        f"- {REPS} reps per scenario",
        f"- market_size_fixed = True, α = 0 (no pooling)",
        f"- log_family = 'laplace', IQR ∈ [0.1, 0.3]",
        f"- metric_every = {METRIC_EVERY}",
        "",
        "## Calibration Result",
        "",
    ]

    if c_star is not None:
        lines.extend([
            f"**c_star = {c_star:.6f}**",
            "",
            f"Hill exponent at c_star = {hill_at_c_star:.4f} (target = {TARGET_HILL})",
            "",
        ])
    else:
        lines.extend([
            f"**Target Hill = {TARGET_HILL} not reached on grid.**",
            "",
            f"Maximum Hill achieved = {hill_at_c_star:.4f}",
            "",
        ])

    # Table for N=50
    lines.extend([
        "## N=50 Results",
        "",
        "| c | 1/(1-c) | Hill [25,75] | HHI [25,75] | Top10 | Floor frac |",
        "|---|---------|--------------|-------------|-------|------------|",
    ])

    for r in sorted(n50_results, key=lambda x: x['c']):
        lines.append(
            f"| {r['c']:.4f} | {r['theoretical_hill']:.4f} | "
            f"{r['hill_median']:.4f} [{r['hill_p25']:.4f}, {r['hill_p75']:.4f}] | "
            f"{r['hhi_median']:.4f} [{r['hhi_p25']:.4f}, {r['hhi_p75']:.4f}] | "
            f"{r['top10_median']:.4f} | {r['floor_frac_median']:.4f} |"
        )

    # Table for N=200
    if n200_results:
        lines.extend([
            "",
            "## N=200 Results (finite-N correction check)",
            "",
            "| c | 1/(1-c) | Hill [25,75] | HHI [25,75] | Top10 | Floor frac |",
            "|---|---------|--------------|-------------|-------|------------|",
        ])

        for r in sorted(n200_results, key=lambda x: x['c']):
            lines.append(
                f"| {r['c']:.4f} | {r['theoretical_hill']:.4f} | "
                f"{r['hill_median']:.4f} [{r['hill_p25']:.4f}, {r['hill_p75']:.4f}] | "
                f"{r['hhi_median']:.4f} [{r['hhi_p25']:.4f}, {r['hhi_p75']:.4f}] | "
                f"{r['top10_median']:.4f} | {r['floor_frac_median']:.4f} |"
            )

    # Figure description
    lines.extend([
        "",
        "## Figure",
        "",
        "![Hill exponent vs c](floor_calibration.png)",
        "",
        "The figure shows Hill exponent vs floor coefficient c for N=50 and N=200,",
        "with the theoretical 1/(1-c) curve for comparison.",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Wrote {output_path}")


def create_figure(aggregated, c_star, output_path):
    """Create Hill vs c figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n50_results = sorted([r for r in aggregated if r['N'] == 50], key=lambda x: x['c'])
    n200_results = sorted([r for r in aggregated if r['N'] == 200], key=lambda x: x['c'])

    fig, ax = plt.subplots(figsize=(8, 5))

    # N=50 data
    c_50 = [r['c'] for r in n50_results]
    hill_50 = [r['hill_median'] for r in n50_results]
    hill_50_lo = [r['hill_p25'] for r in n50_results]
    hill_50_hi = [r['hill_p75'] for r in n50_results]

    ax.plot(c_50, hill_50, 'o-', label='N=50', color='C0')
    ax.fill_between(c_50, hill_50_lo, hill_50_hi, alpha=0.2, color='C0')

    # N=200 data
    if n200_results:
        c_200 = [r['c'] for r in n200_results]
        hill_200 = [r['hill_median'] for r in n200_results]
        hill_200_lo = [r['hill_p25'] for r in n200_results]
        hill_200_hi = [r['hill_p75'] for r in n200_results]

        ax.plot(c_200, hill_200, 's-', label='N=200', color='C1')
        ax.fill_between(c_200, hill_200_lo, hill_200_hi, alpha=0.2, color='C1')

    # Theoretical curve 1/(1-c)
    c_theory = np.linspace(0.01, 0.55, 100)
    hill_theory = 1.0 / (1.0 - c_theory)
    ax.plot(c_theory, hill_theory, '--', label='1/(1-c)', color='gray')

    # Target line
    ax.axhline(TARGET_HILL, color='red', linestyle=':', label=f'Target = {TARGET_HILL}')

    # c_star marker
    if c_star is not None:
        ax.axvline(c_star, color='green', linestyle=':', alpha=0.7)
        ax.annotate(f'c* = {c_star:.4f}', xy=(c_star, TARGET_HILL),
                    xytext=(c_star + 0.05, TARGET_HILL + 0.1),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    color='green', fontsize=10)

    ax.set_xlabel('Floor coefficient c')
    ax.set_ylabel('Hill exponent')
    ax.set_title('Hill Exponent vs Floor Coefficient')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0.8, 2.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Wrote {output_path}")


def update_scenarios_json(c_star, scenarios_path):
    """
    Update pilot_c/scenarios.json:
    1. Set floor_c = c_star for every scenario
    2. Add floor-level block with c_star×0.5 and c_star×2 (laplace, power_law, 9 α, 5 reps)
    """
    with open(scenarios_path, 'r') as f:
        scenarios = json.load(f)

    # Find max scenario_id
    max_id = max(s['scenario_id'] for s in scenarios)
    new_id = max_id + 1

    # Update floor_c for all existing scenarios
    for s in scenarios:
        s['floor_c'] = c_star

    # Template for new scenarios (copy from first main block scenario)
    template = None
    for s in scenarios:
        if s['block'] == 'main':
            template = s.copy()
            break

    if template is None:
        raise ValueError("No main block scenario found")

    # floor-level block parameters
    floor_levels = [c_star * 0.5, c_star * 2.0]
    alphas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # 9 alpha values
    families = ['laplace']
    cost_types = ['power_law']
    reps_per = 5

    new_scenarios = []
    cell_id = 0

    for floor_c in floor_levels:
        for family in families:
            for cost_type in cost_types:
                for alpha in alphas:
                    for rep in range(reps_per):
                        new_s = template.copy()
                        new_s['block'] = 'floor-level'
                        new_s['scenario_id'] = new_id
                        new_s['cell_id'] = cell_id
                        new_s['log_family'] = family
                        new_s['cost_type'] = cost_type
                        new_s['alpha'] = alpha
                        new_s['floor_c'] = floor_c
                        new_s['rep'] = rep
                        new_s['seed'] = SEED_BASE + new_id
                        new_scenarios.append(new_s)
                        new_id += 1
                    cell_id += 1

    # Append new scenarios
    scenarios.extend(new_scenarios)

    with open(scenarios_path, 'w') as f:
        json.dump(scenarios, f, indent=2)

    print(f"Updated {scenarios_path}: set floor_c={c_star:.6f} for all scenarios, "
          f"added {len(new_scenarios)} floor-level scenarios")

    return len(new_scenarios)


def main():
    """Main calibration routine."""
    print("=" * 60)
    print("C13: Floor Calibration")
    print("=" * 60)

    analytics_dir = Path(__file__).parent

    all_results = []

    # Run N=50 grid
    print(f"\nN=50 grid: c ∈ {C_GRID_N50}")
    total_runs_n50 = len(C_GRID_N50) * REPS
    run_idx = 0

    for c in C_GRID_N50:
        for rep in range(REPS):
            seed = SEED_BASE + run_idx
            run_idx += 1
            print(f"  [{run_idx}/{total_runs_n50}] N=50, c={c:.4f}, rep={rep}, seed={seed}...")
            result = run_calibration_scenario(N=50, c=c, rep=rep, seed=seed)
            print(f"    Hill={result['hill_median']:.4f}, HHI={result['hhi_within_median']:.4f}")
            all_results.append(result)

    # Run N=200 subset
    print(f"\nN=200 subset: c ∈ {C_GRID_N200}")
    total_runs_n200 = len(C_GRID_N200) * REPS
    run_idx = 0

    for c in C_GRID_N200:
        for rep in range(REPS):
            seed = SEED_BASE + 1000 + run_idx  # Different seed range
            run_idx += 1
            print(f"  [{run_idx}/{total_runs_n200}] N=200, c={c:.4f}, rep={rep}, seed={seed}...")
            result = run_calibration_scenario(N=200, c=c, rep=rep, seed=seed)
            print(f"    Hill={result['hill_median']:.4f}, HHI={result['hhi_within_median']:.4f}")
            all_results.append(result)

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(all_results)

    # Find c_star
    c_star, hill_at_c_star = find_c_star(aggregated, N=50)

    if c_star is not None:
        print(f"\nc_star = {c_star:.6f} (Hill = {hill_at_c_star:.4f})")
    else:
        print(f"\nTarget Hill = {TARGET_HILL} not reached. Max Hill = {hill_at_c_star:.4f}")
        print("STOPPING: Cannot proceed without c_star.")
        return 1

    # Write outputs
    write_csv(aggregated, analytics_dir / 'floor_calibration.csv')
    write_markdown(aggregated, c_star, hill_at_c_star, analytics_dir / 'floor_calibration.md')
    create_figure(aggregated, c_star, analytics_dir / 'floor_calibration.png')

    # Update scenarios.json
    scenarios_path = analytics_dir.parent / 'pilot_c' / 'scenarios.json'
    if scenarios_path.exists():
        n_added = update_scenarios_json(c_star, scenarios_path)
        print(f"\nAdded {n_added} floor-level scenarios to pilot_c/scenarios.json")
    else:
        print(f"\nWARNING: {scenarios_path} not found, skipping update")

    print("\n" + "=" * 60)
    print("Calibration complete")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
