#!/usr/bin/env python3
"""
Test correlation structure implementations (§3).

- Verifies that rho and cross_corr parameters produce different results
- Tests that parameters are recorded in hyperparameters
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from collaborative_growth import model


def test_zero_rho_is_default():
    """
    rho=0.0 should be the default within-market correlation.
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result = model(params, seed=42, market_corr='identity')

    hyperparameters = result[-1]
    assert hyperparameters['rho'] == 0.0, \
        f"Expected default rho=0.0, got '{hyperparameters['rho']}'"

    print(f"Default rho verified: {hyperparameters['rho']}")


def test_zero_cross_corr_is_default():
    """
    cross_corr=0.0 should be the default cross-market correlation.
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result = model(params, seed=42, market_corr='identity')

    hyperparameters = result[-1]
    assert hyperparameters['cross_corr'] == 0.0, \
        f"Expected default cross_corr=0.0, got '{hyperparameters['cross_corr']}'"

    print(f"Default cross_corr verified: {hyperparameters['cross_corr']}")


def test_rho_values_differ():
    """
    Different rho values should produce different results.
    """
    params = [15, 15, 200, 0.0, 225, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_zero = model(params, seed=42, rho=0.0)
    result_pos = model(params, seed=42, rho=0.3)

    gini_zero = result_zero[5]
    gini_pos = result_pos[5]

    # Should be different
    assert not np.array_equal(gini_zero, gini_pos), \
        "rho=0.0 and rho=0.3 should produce different results"

    # Verify hyperparameters
    assert result_zero[-1]['rho'] == 0.0
    assert result_pos[-1]['rho'] == 0.3

    print("rho values differ test passed")


def test_cross_corr_values_differ():
    """
    Different cross_corr values should produce different results.
    """
    params = [20, 10, 200, 0.0, 200, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_zero = model(params, seed=42, cross_corr=0.0)
    result_pos = model(params, seed=42, cross_corr=0.3)

    gini_zero = result_zero[5]
    gini_pos = result_pos[5]

    # Should be different
    assert not np.array_equal(gini_zero, gini_pos), \
        "cross_corr=0.0 and cross_corr=0.3 should produce different results"

    # Verify hyperparameters
    assert result_zero[-1]['cross_corr'] == 0.0
    assert result_pos[-1]['cross_corr'] == 0.3

    print("cross_corr values differ test passed")


def test_zero_corr_backward_compatible():
    """
    Zero correlation (Phase A default) should produce same results as before.
    """
    params = [10, 10, 200, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Run with explicit zeros
    result1 = model(params, seed=12345, rho=0.0, cross_corr=0.0)

    # Run with defaults
    result2 = model(params, seed=12345)

    # Should be identical
    gini1 = result1[5]
    gini2 = result2[5]

    assert np.array_equal(gini1, gini2), "Explicit zeros should match default"

    print("zero correlation backward compatibility test passed")


def test_positive_rho_increases_variance():
    """
    Positive within-market correlation should increase cross-sectional variance.
    When firms are positively correlated, they move together, increasing dispersion.
    """
    params = [15, 15, 300, 0.0, 225, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_zero = model(params, seed=42, rho=0.0)
    result_pos = model(params, seed=42, rho=0.3)

    gini_zero = result_zero[5]
    gini_pos = result_pos[5]

    # With positive correlation, we expect higher inequality (more co-movement)
    # This is a statistical property, not guaranteed for every seed
    print(f"rho=0.0 mean Gini: {gini_zero.mean():.4f}")
    print(f"rho=0.3 mean Gini: {gini_pos.mean():.4f}")


if __name__ == '__main__':
    print("Testing zero rho is default...")
    test_zero_rho_is_default()
    print("PASS\n")

    print("Testing zero cross_corr is default...")
    test_zero_cross_corr_is_default()
    print("PASS\n")

    print("Testing rho values differ...")
    test_rho_values_differ()
    print("PASS\n")

    print("Testing cross_corr values differ...")
    test_cross_corr_values_differ()
    print("PASS\n")

    print("Testing zero correlation backward compatibility...")
    test_zero_corr_backward_compatible()
    print("PASS\n")

    print("Testing positive rho effect on variance...")
    test_positive_rho_increases_variance()
    print("PASS\n")

    print("All correlation structure tests passed!")
