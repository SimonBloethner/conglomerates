#!/usr/bin/env python3
"""
Test sharing rule implementations (§2).

- Verifies that ewp and cap pooling rules produce different results
- Tests that pool_history and pool_window parameters are recorded
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from collaborative_growth import model


def test_ewp_is_default():
    """
    ewp (equal-weight pooling) should be the default pooling rule.
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result = model(params, seed=42, market_corr='identity')

    hyperparameters = result[-1]
    assert hyperparameters['pooling_rule'] == 'ewp', \
        f"Expected default pooling_rule='ewp', got '{hyperparameters['pooling_rule']}'"

    print(f"Default pooling_rule verified: {hyperparameters['pooling_rule']}")


def test_ewp_vs_cap_differ():
    """
    ewp and cap pooling rules should produce different results.
    """
    params = [15, 15, 300, 0.15, 225, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_ewp = model(params, seed=42, market_corr='identity',
                       pooling_rule='ewp')
    result_cap = model(params, seed=42, market_corr='identity',
                       pooling_rule='cap')

    gini_ewp = result_ewp[5]
    gini_cap = result_cap[5]

    # Should be different (different pooling rules)
    assert not np.array_equal(gini_ewp, gini_cap), \
        "ewp and cap should produce different Gini trajectories"

    # Verify hyperparameters captured the rules
    assert result_ewp[-1]['pooling_rule'] == 'ewp'
    assert result_cap[-1]['pooling_rule'] == 'cap'

    print(f"ewp vs cap differ test passed")
    # Handle multi-dimensional gini arrays (average across markets if needed)
    ewp_final = np.mean(gini_ewp[-1]) if gini_ewp[-1].ndim > 0 else float(gini_ewp[-1])
    cap_final = np.mean(gini_cap[-1]) if gini_cap[-1].ndim > 0 else float(gini_cap[-1])
    print(f"  ewp final Gini: {ewp_final:.4f}")
    print(f"  cap final Gini: {cap_final:.4f}")


def test_ewp_backward_compatible():
    """
    ewp (Phase A default) should produce same results as before.
    """
    params = [10, 10, 200, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Run with explicit ewp
    result1 = model(params, seed=12345, market_corr='identity',
                    pooling_rule='ewp')

    # Run with default (should be ewp)
    result2 = model(params, seed=12345, market_corr='identity')

    # Should be identical
    gini1 = result1[5]
    gini2 = result2[5]

    assert np.array_equal(gini1, gini2), "Explicit ewp should match default"

    print("ewp backward compatibility test passed")


def test_pool_window_recorded():
    """
    pool_window parameter should be recorded in hyperparameters.
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # With explicit pool_window
    result = model(params, seed=42, market_corr='identity',
                   pool_window=25)

    hyperparameters = result[-1]
    assert hyperparameters['pool_window'] == 25, \
        f"Expected pool_window=25, got {hyperparameters['pool_window']}"

    print(f"pool_window recorded: {hyperparameters['pool_window']}")


def test_pool_window_defaults_to_lookback():
    """
    pool_window should default to lookback value when not specified.
    """
    lookback = 50
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, lookback,
              'power_law', None, None, None]

    result = model(params, seed=42, market_corr='identity')

    hyperparameters = result[-1]
    assert hyperparameters['pool_window'] == lookback, \
        f"Expected pool_window={lookback}, got {hyperparameters['pool_window']}"

    print(f"pool_window defaults to lookback: {hyperparameters['pool_window']}")


def test_pool_history_recorded():
    """
    pool_history parameter should be recorded in hyperparameters.
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_rolling = model(params, seed=42, pool_history='rolling')
    result_full = model(params, seed=42, pool_history='full')

    assert result_rolling[-1]['pool_history'] == 'rolling'
    assert result_full[-1]['pool_history'] == 'full'

    print("pool_history recording test passed")


if __name__ == '__main__':
    print("Testing ewp is default...")
    test_ewp_is_default()
    print("PASS\n")

    print("Testing ewp vs cap differ...")
    test_ewp_vs_cap_differ()
    print("PASS\n")

    print("Testing ewp backward compatibility...")
    test_ewp_backward_compatible()
    print("PASS\n")

    print("Testing pool_window recorded...")
    test_pool_window_recorded()
    print("PASS\n")

    print("Testing pool_window defaults to lookback...")
    test_pool_window_defaults_to_lookback()
    print("PASS\n")

    print("Testing pool_history recorded...")
    test_pool_history_recorded()
    print("PASS\n")

    print("All sharing rule tests passed!")
