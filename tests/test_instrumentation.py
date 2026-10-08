#!/usr/bin/env python3
"""
C17: Tests for instrumentation features.

Tests:
1. Per-type acceptance rate tracking (ss, sc, cc)
2. Member-standalone growth gap metric
3. K_mean in summary
4. top10pct_aggregate rename
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collaborative_growth import model


def make_test_scenario():
    """Create a minimal test scenario."""
    return {
        'M': 5,
        'N': 10,
        'T': 200,
        'burn_in': 100,
        'merge_thresh': 0.05,
        'proportional': False,
        'growth_process': 'log_family',
        'log_family': 'laplace',
        'mu_range': [0.01, 0.1],
        'sigma_range': [0.1, 0.3],
        'floor_c': 0.12717,
        'metric_every': 50,
        'market_size_fixed': True,
        'cost_type': 'power_law',
        'c0': 0.00002032,
        'c1': 1.2,
        'c2': 0.001,
        'lookback': 20,
        'cross_corr': 0.0,
        'alpha': 0.3,  # High enough to trigger mergers
        'sharing_rule': 'proportional',
        'alpha_endogenous': False,
        'decision_rule': 'replay',
        'seed': 42,
    }


def run_scenario(scenario):
    """Run a scenario and return results."""
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
        mu_range=tuple(scenario['mu_range']),
        sigma_range=tuple(scenario['sigma_range']),
        floor_c=scenario['floor_c'],
        cross_corr=scenario['cross_corr'],
        metric_every=scenario['metric_every'],
        sharing_rule=scenario['sharing_rule'],
        burn_in=scenario['burn_in'],
        alpha_endogenous=scenario.get('alpha_endogenous', False),
        market_size_fixed=scenario.get('market_size_fixed', False),
    )

    return result


def test_per_type_acceptance_arrays_exist():
    """
    Test that per-type proposal/merger arrays exist in hyperparameters.
    """
    scenario = make_test_scenario()
    results = run_scenario(scenario)
    hyperparams = results[-1]

    # Check arrays exist
    assert 'proposals_ss_per_period' in hyperparams, "Missing proposals_ss_per_period"
    assert 'proposals_sc_per_period' in hyperparams, "Missing proposals_sc_per_period"
    assert 'proposals_cc_per_period' in hyperparams, "Missing proposals_cc_per_period"
    assert 'mergers_ss_per_period' in hyperparams, "Missing mergers_ss_per_period"
    assert 'mergers_sc_per_period' in hyperparams, "Missing mergers_sc_per_period"
    assert 'mergers_cc_per_period' in hyperparams, "Missing mergers_cc_per_period"

    # Check shape (should match T)
    T = scenario['T']
    assert hyperparams['proposals_ss_per_period'].shape == (T,), \
        f"Wrong shape: {hyperparams['proposals_ss_per_period'].shape}"

    print("PASS: Per-type arrays exist with correct shape")


def test_per_type_acceptance_rates_in_summary():
    """
    Test that per-type acceptance rates exist in summary.
    """
    scenario = make_test_scenario()
    results = run_scenario(scenario)
    hyperparams = results[-1]
    summary = hyperparams['summary']

    # Check rates exist
    assert 'acceptance_rate_ss' in summary, "Missing acceptance_rate_ss"
    assert 'acceptance_rate_sc' in summary, "Missing acceptance_rate_sc"
    assert 'acceptance_rate_cc' in summary, "Missing acceptance_rate_cc"

    # Check per-period counts exist
    assert 'proposals_ss_per_period' in summary, "Missing proposals_ss_per_period in summary"
    assert 'mergers_ss_per_period' in summary, "Missing mergers_ss_per_period in summary"

    print("PASS: Per-type acceptance rates exist in summary")


def test_per_type_counts_sum_to_total():
    """
    Test that per-type proposal counts sum to total proposals.
    """
    scenario = make_test_scenario()
    results = run_scenario(scenario)
    hyperparams = results[-1]
    proposals_per_period = results[8]

    # Get per-type arrays
    props_ss = hyperparams['proposals_ss_per_period']
    props_sc = hyperparams['proposals_sc_per_period']
    props_cc = hyperparams['proposals_cc_per_period']

    # Sum per-type should equal total
    per_type_sum = props_ss + props_sc + props_cc
    np.testing.assert_array_almost_equal(
        per_type_sum, proposals_per_period,
        decimal=10,
        err_msg="Per-type proposals don't sum to total"
    )

    print("PASS: Per-type proposals sum to total")


def test_growth_gap_arrays_exist():
    """
    Test that growth gap arrays exist in hyperparameters.
    """
    scenario = make_test_scenario()
    results = run_scenario(scenario)
    hyperparams = results[-1]

    assert 'growth_gap' in hyperparams, "Missing growth_gap array"
    assert 'growth_gap_by_market' in hyperparams, "Missing growth_gap_by_market array"

    # Check shapes
    n_metric_steps = scenario['T'] // scenario['metric_every']
    assert hyperparams['growth_gap'].shape == (n_metric_steps,), \
        f"Wrong growth_gap shape: {hyperparams['growth_gap'].shape}"
    assert hyperparams['growth_gap_by_market'].shape == (scenario['M'], n_metric_steps), \
        f"Wrong growth_gap_by_market shape: {hyperparams['growth_gap_by_market'].shape}"

    print("PASS: Growth gap arrays exist with correct shape")


def test_growth_gap_median_in_summary():
    """
    Test that growth_gap_median exists in summary.
    """
    scenario = make_test_scenario()
    results = run_scenario(scenario)
    hyperparams = results[-1]
    summary = hyperparams['summary']

    assert 'growth_gap_median' in summary, "Missing growth_gap_median in summary"

    # Check it's a scalar (not NaN if there are conglomerates)
    gap_median = summary['growth_gap_median']
    print(f"  growth_gap_median = {gap_median}")

    print("PASS: growth_gap_median exists in summary")


def test_k_mean_in_summary():
    """
    Test that K_mean exists in summary alongside K_median.
    """
    scenario = make_test_scenario()
    results = run_scenario(scenario)
    hyperparams = results[-1]
    summary = hyperparams['summary']

    assert 'K_median' in summary, "Missing K_median in summary"
    assert 'K_mean' in summary, "Missing K_mean in summary"

    print(f"  K_median = {summary['K_median']}")
    print(f"  K_mean = {summary['K_mean']}")

    # If K_median is not NaN, K_mean should also not be NaN
    if not np.isnan(summary['K_median']):
        assert not np.isnan(summary['K_mean']), "K_mean is NaN when K_median is not"

    print("PASS: K_mean exists in summary")


def test_top10pct_aggregate_rename():
    """
    Test that top10pct_aggregate is used (not top10_aggregate).
    """
    scenario = make_test_scenario()
    results = run_scenario(scenario)
    hyperparams = results[-1]
    summary = hyperparams['summary']

    # Check new name exists
    assert 'top10pct_aggregate' in hyperparams, "Missing top10pct_aggregate in hyperparams"
    assert 'top10pct_aggregate_median' in summary, "Missing top10pct_aggregate_median in summary"

    # Check old name doesn't exist
    assert 'top10_aggregate' not in hyperparams, "Old name top10_aggregate still exists"
    assert 'top10_aggregate_median' not in summary, "Old name top10_aggregate_median still exists"

    print("PASS: top10pct_aggregate rename successful")


if __name__ == '__main__':
    print("=" * 60)
    print("C17: Instrumentation Tests")
    print("=" * 60)

    tests = [
        ("Per-type arrays exist", test_per_type_acceptance_arrays_exist),
        ("Per-type acceptance rates in summary", test_per_type_acceptance_rates_in_summary),
        ("Per-type counts sum to total", test_per_type_counts_sum_to_total),
        ("Growth gap arrays exist", test_growth_gap_arrays_exist),
        ("Growth gap median in summary", test_growth_gap_median_in_summary),
        ("K_mean in summary", test_k_mean_in_summary),
        ("top10pct_aggregate rename", test_top10pct_aggregate_rename),
    ]

    failed = 0
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        print("-" * 40)
        try:
            test_fn()
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("All instrumentation tests PASSED")
    else:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
    print("=" * 60)
