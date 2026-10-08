#!/usr/bin/env python3
"""
Phase C pilot runner.

Runs a single scenario from pilot_c/scenarios.json by index.
Saves result to pilot_c/results/scenario_{idx:04d}.pkl.gz (gzip compressed)

Usage:
    python run_pilot_c.py <scenario_index>
"""
import gzip
import json
import pickle
import sys
import time
from pathlib import Path

from collaborative_growth import model, seed_numba


def run_scenario(scenario):
    """Run a single scenario and return results."""
    start_time = time.time()

    # Build params list for model()
    total_firms = scenario['M'] * scenario['N']
    params = [
        scenario['M'],
        scenario['N'],
        scenario['T'],
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
        mu_range=tuple(scenario['mu_range']) if scenario.get('mu_range') else (0.01, 0.1),
        sigma_range=tuple(scenario['sigma_range']),
        floor_c=scenario['floor_c'],
        cross_corr=scenario['cross_corr'],
        metric_every=scenario['metric_every'],
        sharing_rule=scenario['sharing_rule'],
        burn_in=scenario['burn_in'],
        alpha_endogenous=scenario.get('alpha_endogenous', False),
        g=scenario.get('g'),
        market_size_fixed=scenario.get('market_size_fixed', False),
        decision_rule=scenario.get('decision_rule', 'loggain'),  # C18: default loggain
        exit_review_every=scenario.get('exit_review_every', 5),  # C21: default 5
    )

    elapsed = time.time() - start_time

    # Extract hyperparameters dict (last element)
    hyperparams = result[-1]

    # Build output dict with scenario metadata and results
    output = {
        'scenario': scenario,
        'elapsed_seconds': elapsed,
        'ms_per_step': 1000 * elapsed / scenario['T'],
        # Time series metrics
        'hill_exponent': hyperparams.get('hill_exponent'),
        'hhi_within': hyperparams.get('hhi_within'),
        'hhi_aggregate': hyperparams.get('hhi_aggregate'),
        'top10pct_aggregate': hyperparams.get('top10pct_aggregate'),
        'cong_capital_share': hyperparams.get('cong_capital_share'),
        'effective_members': hyperparams.get('effective_members'),
        'floor_hits_by_status': hyperparams.get('floor_hits_by_status'),
        'alpha_history': hyperparams.get('alpha_history'),
        # Summary scalars
        'summary': hyperparams.get('summary'),
        # Assortativity data (C11)
        'market_iqr': hyperparams.get('market_iqr'),
        'final_firm_conglom': hyperparams.get('final_firm_conglom'),
    }

    return output


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_pilot_c.py <scenario_index>")
        sys.exit(1)

    idx = int(sys.argv[1])

    # Load scenarios
    with open('pilot_c/scenarios.json', 'r') as f:
        scenarios = json.load(f)

    if idx < 0 or idx >= len(scenarios):
        print(f"Error: scenario index {idx} out of range [0, {len(scenarios)})")
        sys.exit(1)

    scenario = scenarios[idx]
    print(f"Running scenario {idx}: block={scenario['block']}, "
          f"family={scenario['log_family']}, cost={scenario['cost_type']}, "
          f"alpha={scenario['alpha']}, rep={scenario['rep']}")

    # Run
    result = run_scenario(scenario)

    # Save with gzip compression (workaround for NFS 1MB write limit)
    output_dir = Path('pilot_c/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scenario_{idx:04d}.pkl.gz"

    with gzip.open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved to {output_path}")
    print(f"Runtime: {result['elapsed_seconds']:.1f}s ({result['ms_per_step']:.2f} ms/step)")


if __name__ == '__main__':
    main()
