#!/usr/bin/env python3
"""
Phase C pilot runner.

Runs a single scenario from pilot_c/scenarios.json by index.
Saves result to pilot_c/results/scenario_{idx:04d}.pkl.gz (gzip compressed)

Usage:
    python run_pilot_c.py <scenario_index> [--out DIR] [--pilot c|d]
--out (C35): write to DIR instead of pilot_c/results (the C35 reruns go to pilot_c/results_c35).
--pilot (C37): read pilot_<p>/scenarios.json and write to pilot_<p>/results by default (c: pilot C, the default;
d: pilot D, the moment rule).
"""
import gzip
import json
import pickle
import sys
import time
from pathlib import Path

from collaborative_growth import model, seed_numba
from pilots import pop_pilot, pilot_dir


def run_scenario(scenario):
    """Run a single scenario and return results."""
    start_time = time.time()

    # C34: scenario keys (the stored schema, unchanged) -> model() keyword names (paper notation)
    assert not scenario['proportional'], "cost as a fixed amount was removed in C34"
    result = model(
        M=scenario['M'],
        N=scenario['N'],
        T=scenario['T'],
        alpha=scenario['alpha'],
        search_prob=scenario['merge_thresh'],
        window=scenario['lookback'],
        cost_form=scenario['cost_type'],
        b0=scenario['c0'],
        b1=scenario['c1'],
        b2=scenario['c2'],
        cost_as_drag=True,
        seed=scenario['seed'],
        growth_process=scenario['growth_process'],
        family=scenario['log_family'],
        nu=scenario.get('nu') or 3.0,  # C37: Student-t degrees of freedom (pilot D's nu block); 3.0 is the default
        mu_range=tuple(scenario['mu_range']) if scenario.get('mu_range') else (0.01, 0.1),
        iqr_range=tuple(scenario['sigma_range']),
        floor_c=scenario['floor_c'],
        cross_corr=scenario['cross_corr'],
        metric_every=scenario['metric_every'],
        sharing_rule=scenario['sharing_rule'],
        burn_in=scenario['burn_in'],
        alpha_endogenous=scenario.get('alpha_endogenous', False),
        g=scenario.get('g'),
        market_size_fixed=scenario.get('market_size_fixed', False),
        decision_rule=scenario.get('decision_rule', 'loggain'),  # C18: default loggain
    )

    elapsed = time.time() - start_time

    # C34: a unit in which every member is capped (a wipeout) was expected never to occur in the pilot. It does
    # once (scenario 438: two t3 members both lose ~100% in the same period), so it is reported, not asserted;
    # the count is in summary['pool_wipeout_total'] and the tidy column of the same name (MODEL_FLAGS.md).
    wipeouts = result[-1]['summary']['pool_wipeout_total']
    if wipeouts:
        print(f"WARNING: {wipeouts:g} pool wipeout(s) in scenario {scenario.get('scenario_id')}", file=sys.stderr)

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
        # C22b: alpha scatter for endogenous alpha
        'alpha_scatter': hyperparams.get('alpha_scatter'),
        'alpha_scatter_final': hyperparams.get('alpha_scatter_final'),
        # C35: proposals and mergers by K = size of the merged set, binding-gain histogram (diagnostics)
        'proposals_by_K': hyperparams.get('proposals_by_K'),
        'mergers_by_K': hyperparams.get('mergers_by_K'),
        'proposals_by_K_per_period': hyperparams.get('proposals_by_K_per_period'),
        'mergers_by_K_per_period': hyperparams.get('mergers_by_K_per_period'),
        'binding_gain_hist_by_K': hyperparams.get('binding_gain_hist_by_K'),
        'binding_gain_edges': hyperparams.get('binding_gain_edges'),
        'proposals_ss_per_period': hyperparams.get('proposals_ss_per_period'),
        'proposals_sc_per_period': hyperparams.get('proposals_sc_per_period'),
        'proposals_cc_per_period': hyperparams.get('proposals_cc_per_period'),
        'mergers_ss_per_period': hyperparams.get('mergers_ss_per_period'),
        'mergers_sc_per_period': hyperparams.get('mergers_sc_per_period'),
        'mergers_cc_per_period': hyperparams.get('mergers_cc_per_period'),
    }

    return output


def main():
    pilot = pop_pilot()
    args = sys.argv[1:]
    out_dir = pilot_dir(pilot, 'results')
    if '--out' in args:
        k = args.index('--out')
        out_dir = args[k + 1]
        del args[k:k + 2]
    if len(args) != 1:
        print("Usage: python run_pilot_c.py <scenario_index> [--out DIR] [--pilot c|d]")
        sys.exit(1)

    idx = int(args[0])

    # Load scenarios
    with open(pilot_dir(pilot, 'scenarios.json'), 'r') as f:
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
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scenario_{idx:04d}.pkl.gz"

    with gzip.open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved to {output_path}")
    print(f"Runtime: {result['elapsed_seconds']:.1f}s ({result['ms_per_step']:.2f} ms/step)")


if __name__ == '__main__':
    main()
