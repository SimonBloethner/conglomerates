"""
Test: Assortativity by IQR (C22c)

Tests that assort_iqr (Pearson correlation of per-firm IQR with
mean IQR of same conglomerate) is computed correctly and has
sensible values.
"""

import numpy as np
import sys
sys.path.insert(0, '/groups/m-larch/bt307958/IOxEE')
import collaborative_growth as cg


def test_assortativity_in_range():
    """
    Test: assort_iqr is in valid correlation range [-1, 1].

    Test catches: computation errors producing invalid correlations.
    """
    np.random.seed(42)
    cg.seed_numba(42)

    M = N = 20
    T = 600
    alpha = 0.3
    lookback = 50

    params = [
        M, N, T, alpha, M * N,
        0.1, 4, 0.0, False, lookback,
        'power_law', None, None, None,
    ]

    result = cg.model(
        params, seed=42,
        growth_process='log_family', log_family='laplace',
        sigma_range=(0.1, 0.3), floor_c=0.12717,
        g=0.02, market_size_fixed=True,
        decision_rule='loggain'
    )

    hyper = result[-1]
    summary = hyper['summary']

    # Check assort_iqr is present
    assert 'assort_iqr' in summary, "assort_iqr not in summary"
    assert 'assort_iqr' in hyper, "assort_iqr not in hyperparameters"

    assort_iqr = hyper['assort_iqr']

    # Should be in [-1, 1] or NaN (if no conglomerates)
    if not np.isnan(assort_iqr):
        assert -1.0 <= assort_iqr <= 1.0, f"assort_iqr {assort_iqr} out of range [-1, 1]"
        print(f"assort_iqr = {assort_iqr:.4f}")
    else:
        print("assort_iqr is NaN (no conglomerates with K>=2)")

    print("PASS: assort_iqr is in valid range")


def test_assortativity_random_assignment_near_zero():
    """
    Test: For random conglomerate assignment, assort_iqr should be near zero.

    This is because random assignment should not correlate with IQR.
    We run the model with alpha=0 (no mergers) then manually assign
    random conglomerates and recompute assortativity.

    Test catches: assortativity computation being spuriously high.
    """
    np.random.seed(42)
    cg.seed_numba(42)

    M = N = 20
    T = 100  # Short run
    alpha = 0.0  # No mergers
    lookback = 50

    params = [
        M, N, T, alpha, M * N,
        0.1, 4, 0.0, False, lookback,
        'power_law', None, None, None,
    ]

    result = cg.model(
        params, seed=42,
        growth_process='log_family', log_family='laplace',
        sigma_range=(0.1, 0.3), floor_c=0.12717,
        g=0.02, market_size_fixed=True
    )

    hyper = result[-1]
    market_iqr = hyper['market_iqr']
    markets = hyper['markets']
    firms_per_market = hyper['firms_per_market']
    total_firms = markets * firms_per_market

    # Per-firm IQR
    firm_home_market = np.repeat(np.arange(markets), firms_per_market)
    firm_iqr = market_iqr[firm_home_market]

    # Random conglomerate assignment:
    # Assign each firm to a random conglomerate (0 to 9)
    np.random.seed(123)
    n_cong = 10
    firm_cong_random = np.random.randint(0, n_cong, size=total_firms)

    # Compute mean IQR per conglomerate
    cong_mean_iqr = np.zeros(total_firms, dtype=np.float64)
    for cid in range(n_cong):
        members = np.where(firm_cong_random == cid)[0]
        if len(members) >= 2:
            mean_iqr = np.mean(firm_iqr[members])
            cong_mean_iqr[members] = mean_iqr

    # Compute assortativity for random assignment
    # (all firms are in conglomerates in this test)
    x = firm_iqr
    y = cong_mean_iqr
    random_assort = np.corrcoef(x, y)[0, 1]

    print(f"Random assignment assortativity: {random_assort:.4f}")

    # Should be near zero (abs < 0.2 is a reasonable threshold)
    assert abs(random_assort) < 0.2, f"Random assignment assort {random_assort:.4f} should be near zero"

    print("PASS: Random assignment assortativity is near zero")


def test_assortativity_alpha_zero():
    """
    Test: With alpha=0 (no mergers), assort_iqr should be NaN.

    Test catches: assort_iqr being computed when there are no conglomerates.
    """
    np.random.seed(42)
    cg.seed_numba(42)

    M = N = 10
    T = 100
    alpha = 0.0  # No mergers
    lookback = 50

    params = [
        M, N, T, alpha, M * N,
        0.1, 4, 0.0, False, lookback,
        'power_law', None, None, None,
    ]

    result = cg.model(
        params, seed=42,
        growth_process='log_family', log_family='laplace',
        sigma_range=(0.1, 0.3), floor_c=0.12717,
        g=0.02, market_size_fixed=True
    )

    hyper = result[-1]
    assort_iqr = hyper['assort_iqr']

    # With no mergers, there should be no conglomerates -> NaN
    assert np.isnan(assort_iqr), f"Expected NaN with alpha=0, got {assort_iqr}"

    print("PASS: assort_iqr is NaN when alpha=0")


if __name__ == "__main__":
    test_assortativity_in_range()
    test_assortativity_random_assignment_near_zero()
    test_assortativity_alpha_zero()
    print("\nAll assortativity tests passed!")
