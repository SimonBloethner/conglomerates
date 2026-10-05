#!/usr/bin/env python3
"""
Tests for analytics/benchmarks.py.

Test catches:
1. Gaussian pooling_gain test catches wrong objective or wrong scale conversion
2. Non-decreasing test catches sign errors
3. IQR conversion test catches wrong scale formulas
4. Barrier exponent round-trip catches implementation bugs
"""
import numpy as np
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../analytics')

from analytics.benchmarks import (
    pooling_gain, iqr_to_scale, draw_log_shocks,
    barrier_exponent, c_for_exponent
)


def test_gaussian_pooling_gain_formula():
    """
    For Gaussian with alpha=1 (full pooling), the gain at cost=0 should equal
    σ²/2 · (1 - 1/K), which is the variance reduction from averaging.

    This catches: wrong objective function or wrong scale conversion.
    """
    mu = 0.05
    iqr = 0.2
    sigma = iqr_to_scale('normal', iqr)

    # Theoretical gain for full pooling (alpha=1) with Gaussian:
    # E[log(mean_j e^Xj)] - E[log e^X1]
    # For independent Gaussians, this equals σ²/2 · (1 - 1/K)
    # (because log of average of lognormals gains variance reduction)

    for K in [2, 5, 10, 20]:
        theoretical = (sigma ** 2 / 2) * (1 - 1 / K)

        # Use fewer draws for speed in tests but enough for accuracy
        gain, se = pooling_gain(K, alpha=1.0, family='normal',
                                mu=mu, iqr=iqr, cost=0.0,
                                n_draws=500_000, seed=42)

        # Check within 3 standard errors
        assert abs(gain - theoretical) < 3 * se, (
            f"K={K}: gain={gain:.6f}, theoretical={theoretical:.6f}, "
            f"diff={abs(gain - theoretical):.6f}, 3*SE={3*se:.6f}"
        )

    print("PASS: Gaussian pooling_gain matches σ²/2·(1-1/K) within 3 SE")


def test_pooling_gain_nondecreasing_in_K():
    """
    Pooling gain (at cost=0) should be non-decreasing in K for all families.

    This catches: sign errors in the pooling formula.
    """
    mu = 0.05
    iqr = 0.2

    for family in ['normal', 'laplace', 't3']:
        gains = []
        for K in [1, 2, 5, 10, 20]:
            gain, _ = pooling_gain(K, alpha=0.5, family=family,
                                   mu=mu, iqr=iqr, cost=0.0,
                                   n_draws=200_000, seed=123)
            gains.append(gain)

        # Check non-decreasing (allow small tolerance for MC noise)
        for i in range(1, len(gains)):
            # Allow for Monte Carlo noise: gains should not decrease significantly
            assert gains[i] >= gains[i-1] - 0.001, (
                f"{family}: gain decreased from K={[1,2,5,10,20][i-1]} to "
                f"K={[1,2,5,10,20][i]}: {gains[i-1]:.6f} -> {gains[i]:.6f}"
            )

    print("PASS: pooling_gain is non-decreasing in K for all families")


def test_iqr_conversion():
    """
    Empirical IQR of draws from each family should match target IQR within 1%.

    This catches: wrong scale conversion formulas.
    """
    target_iqr = 0.2
    n_draws = 1_000_000
    rng = np.random.default_rng(42)

    for family in ['normal', 'laplace', 't3']:
        # Draw samples (mu=0 so IQR is just from the distribution shape)
        draws = draw_log_shocks(n_draws, family, mu=0.0, iqr=target_iqr, rng=rng)

        # Compute empirical IQR
        q25, q75 = np.percentile(draws, [25, 75])
        empirical_iqr = q75 - q25

        # Check within 1% of target
        relative_error = abs(empirical_iqr - target_iqr) / target_iqr
        assert relative_error < 0.01, (
            f"{family}: empirical IQR={empirical_iqr:.6f}, "
            f"target={target_iqr}, error={100*relative_error:.2f}%"
        )

    print("PASS: IQR conversion correct for all families (within 1%)")


def test_barrier_exponent_round_trip():
    """
    barrier_exponent(c_for_exponent(target)) should equal target.

    This catches: implementation bugs in barrier functions.
    """
    target = 1.06
    c = c_for_exponent(target)
    result = barrier_exponent(c)

    assert abs(result - target) < 1e-10, (
        f"Round-trip failed: target={target}, c={c}, result={result}"
    )

    # Test a few more values
    for target in [1.5, 2.0, 10.0]:
        c = c_for_exponent(target)
        result = barrier_exponent(c)
        assert abs(result - target) < 1e-10, (
            f"Round-trip failed for target={target}"
        )

    print("PASS: barrier_exponent round-trip correct")


if __name__ == '__main__':
    print("Testing Gaussian pooling gain formula...")
    test_gaussian_pooling_gain_formula()

    print("\nTesting pooling gain non-decreasing...")
    test_pooling_gain_nondecreasing_in_K()

    print("\nTesting IQR conversion...")
    test_iqr_conversion()

    print("\nTesting barrier exponent round-trip...")
    test_barrier_exponent_round_trip()

    print("\nAll benchmark tests passed!")
