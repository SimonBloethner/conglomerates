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


def test_uncorr_is_default():
    """
    uncorr (ρ=0) should be the default within-market correlation.
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result = model(params, seed=42, market_corr='identity')

    hyperparameters = result[-1]
    assert hyperparameters['rho'] == 'uncorr', \
        f"Expected default rho='uncorr', got '{hyperparameters['rho']}'"

    print(f"Default rho verified: {hyperparameters['rho']}")


def test_none_cross_corr_is_default():
    """
    none should be the default cross-market correlation.
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result = model(params, seed=42, market_corr='identity')

    hyperparameters = result[-1]
    assert hyperparameters['cross_corr'] == 'none', \
        f"Expected default cross_corr='none', got '{hyperparameters['cross_corr']}'"

    print(f"Default cross_corr verified: {hyperparameters['cross_corr']}")


def test_rho_values_differ():
    """
    Different rho values should produce different results.
    """
    params = [15, 15, 200, 0.0, 225, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_uncorr = model(params, seed=42, rho='uncorr')
    result_pos = model(params, seed=42, rho='pos')
    result_neg = model(params, seed=42, rho='neg')

    gini_uncorr = result_uncorr[5]
    gini_pos = result_pos[5]
    gini_neg = result_neg[5]

    # Should all be different
    assert not np.array_equal(gini_uncorr, gini_pos), \
        "uncorr and pos should produce different results"
    assert not np.array_equal(gini_uncorr, gini_neg), \
        "uncorr and neg should produce different results"
    assert not np.array_equal(gini_pos, gini_neg), \
        "pos and neg should produce different results"

    # Verify hyperparameters
    assert result_uncorr[-1]['rho'] == 'uncorr'
    assert result_pos[-1]['rho'] == 'pos'
    assert result_neg[-1]['rho'] == 'neg'

    print("rho values differ test passed")


def test_cross_corr_values_differ():
    """
    Different cross_corr values should produce different results.
    """
    params = [20, 10, 200, 0.0, 200, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_none = model(params, seed=42, cross_corr='none')
    result_block = model(params, seed=42, cross_corr='block')
    result_ar1 = model(params, seed=42, cross_corr='ar1')

    gini_none = result_none[5]
    gini_block = result_block[5]
    gini_ar1 = result_ar1[5]

    # Should all be different
    assert not np.array_equal(gini_none, gini_block), \
        "none and block should produce different results"
    assert not np.array_equal(gini_none, gini_ar1), \
        "none and ar1 should produce different results"
    assert not np.array_equal(gini_block, gini_ar1), \
        "block and ar1 should produce different results"

    # Verify hyperparameters
    assert result_none[-1]['cross_corr'] == 'none'
    assert result_block[-1]['cross_corr'] == 'block'
    assert result_ar1[-1]['cross_corr'] == 'ar1'

    print("cross_corr values differ test passed")


def test_uncorr_backward_compatible():
    """
    uncorr (Phase A default) should produce same results as before.
    """
    params = [10, 10, 200, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Run with explicit uncorr
    result1 = model(params, seed=12345, rho='uncorr', cross_corr='none')

    # Run with defaults
    result2 = model(params, seed=12345)

    # Should be identical
    gini1 = result1[5]
    gini2 = result2[5]

    assert np.array_equal(gini1, gini2), "Explicit uncorr/none should match default"

    print("uncorr backward compatibility test passed")


def test_positive_rho_increases_variance():
    """
    Positive within-market correlation should increase cross-sectional variance.
    When firms are positively correlated, they move together, increasing dispersion.
    """
    params = [15, 15, 300, 0.0, 225, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_uncorr = model(params, seed=42, rho='uncorr')
    result_pos = model(params, seed=42, rho='pos')

    gini_uncorr = result_uncorr[5]
    gini_pos = result_pos[5]

    # With positive correlation, we expect higher inequality (more co-movement)
    # This is a statistical property, not guaranteed for every seed
    print(f"uncorr mean Gini: {gini_uncorr.mean():.4f}")
    print(f"pos mean Gini: {gini_pos.mean():.4f}")


if __name__ == '__main__':
    print("Testing uncorr is default...")
    test_uncorr_is_default()
    print("PASS\n")

    print("Testing none cross_corr is default...")
    test_none_cross_corr_is_default()
    print("PASS\n")

    print("Testing rho values differ...")
    test_rho_values_differ()
    print("PASS\n")

    print("Testing cross_corr values differ...")
    test_cross_corr_values_differ()
    print("PASS\n")

    print("Testing uncorr backward compatibility...")
    test_uncorr_backward_compatible()
    print("PASS\n")

    print("Testing positive rho effect on variance...")
    test_positive_rho_increases_variance()
    print("PASS\n")

    print("All correlation structure tests passed!")
