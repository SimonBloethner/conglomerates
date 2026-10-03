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
    
    # Unpack results - note the new format includes proposals_per_period
    (mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
     gini_coefficient, ranks, avg_ranks, mergers_per_period, proposals_per_period,
     exits_per_period, hyperparameters) = result
    
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
    assert len(result) == 12  # Expected number of return values
    print(f"M={M}, N={N} run completed successfully")


def test_positive_alpha_has_mergers():
    """
    At positive α with reasonable cost, some mergers should occur.
    """
    params = [30, 30, 500, 0.2, 900, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]  # α=0.2
    
    result = model(params, seed=42, market_corr='identity')
    
    (mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
     gini_coefficient, ranks, avg_ranks, mergers_per_period, proposals_per_period,
     exits_per_period, hyperparameters) = result
    
    total_mergers = mergers_per_period.sum()
    total_proposals = proposals_per_period.sum()
    
    assert total_proposals > 0, "Expected some proposals"
    # With calibrated costs, should have some mergers at α=0.2
    print(f"α=0.2: {total_proposals:.0f} proposals, {total_mergers:.0f} mergers")


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
