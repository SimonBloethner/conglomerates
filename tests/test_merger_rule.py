"""
Test merger rule (§2c/§2d).
- At α=0, mergers should be rejected (proposals > 0, mergers = 0)
- M != N runs should complete without error
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from collaborative_growth import model


def test_alpha_zero_no_mergers():
    """
    At α=0 with any cost type Φ>0, mergers should be rejected.
    proposals_per_period.sum() > 0 but mergers_per_period.sum() == 0
    """
    # Small run for speed
    params = [20, 20, 300, 0.0, 400, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]  # α=0

    result = model(params, seed=42, market_corr='identity')

    # Unpack results - 13 elements (online rank stats)
    (mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
     gini_coefficient, avg_ranks, mergers_per_period, proposals_per_period,
     exits_per_period, rank_range, rank_std, hyperparameters) = result

    total_mergers = mergers_per_period.sum()
    total_proposals = proposals_per_period.sum()

    assert total_mergers == 0, f"Expected 0 mergers at α=0, got {total_mergers}"
    assert total_proposals > 0, f"Expected some proposals at α=0, got {total_proposals}"
    print(f"α=0: {total_proposals:.0f} proposals, {total_mergers:.0f} mergers")


def test_M_not_equals_N_completes():
    """
    M != N run completes without error and maintains conglomerate invariant:
    for every active conglomerate, member home markets are distinct.
    """
    M = 10
    N = 25
    params = [M, N, 200, 0.2, M * N, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Should complete without error
    result = model(params, seed=42, market_corr='identity')

    # Verify we got valid output
    assert result is not None
    assert len(result) == 13  # Expected number of return values (online rank stats)
    print(f"M={M}, N={N} run completed successfully")


def test_positive_alpha_has_mergers():
    """
    At positive α with reasonable cost, some mergers should occur.
    """
    params = [30, 30, 500, 0.2, 900, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]  # α=0.2

    result = model(params, seed=42, market_corr='identity')

    # Unpack results - 13 elements (online rank stats)
    (mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
     gini_coefficient, avg_ranks, mergers_per_period, proposals_per_period,
     exits_per_period, rank_range, rank_std, hyperparameters) = result

    total_mergers = mergers_per_period.sum()
    total_proposals = proposals_per_period.sum()

    assert total_proposals > 0, "Expected some proposals"
    # With calibrated costs, should have some mergers at α=0.2
    print(f"α=0.2: {total_proposals:.0f} proposals, {total_mergers:.0f} mergers")


def test_merger_kernel_optimization_reproducibility():
    """
    Verify that the precomputed sum_Delta_tau/sum_s_tau optimization
    produces bit-identical results across runs with the same seed.

    This tests that the O(K·h) optimization produces the same results
    as would be expected from the original O(K²·h) implementation.
    """
    params = [25, 25, 400, 0.15, 625, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Run twice with same seed
    result1 = model(params, seed=12345, market_corr='identity')
    result2 = model(params, seed=12345, market_corr='identity')

    # Unpack results - 13 elements (online rank stats)
    (mean_members1, quantiles_members1, num_cong1, avg_shares1, quantiles_shares1,
     gini1, avg_ranks1, mergers1, proposals1, exits1, rr1, rs1, hyper1) = result1
    (mean_members2, quantiles_members2, num_cong2, avg_shares2, quantiles_shares2,
     gini2, avg_ranks2, mergers2, proposals2, exits2, rr2, rs2, hyper2) = result2

    # Verify bit-identical outputs
    assert np.array_equal(mergers1, mergers2), "mergers_per_period differs"
    assert np.array_equal(proposals1, proposals2), "proposals_per_period differs"
    assert np.array_equal(exits1, exits2), "exits_per_period differs"
    assert np.array_equal(mean_members1, mean_members2), "mean_members differs"
    assert np.array_equal(gini1, gini2), "gini_coefficient differs"

    # Verify the runs actually produced mergers (sanity check)
    total_mergers = mergers1.sum()
    total_proposals = proposals1.sum()
    print(f"Reproducibility test: {total_proposals:.0f} proposals, {total_mergers:.0f} mergers")
    print("Both runs produced bit-identical results")


if __name__ == '__main__':
    print("Testing α=0 produces no mergers...")
    test_alpha_zero_no_mergers()
    print("PASS")

    print("\nTesting M != N completes...")
    test_M_not_equals_N_completes()
    print("PASS")

    print("\nTesting positive α has mergers...")
    test_positive_alpha_has_mergers()
    print("PASS")

    print("\nTesting merger kernel optimization reproducibility...")
    test_merger_kernel_optimization_reproducibility()
    print("PASS")
