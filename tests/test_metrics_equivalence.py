"""
Test metrics equivalence (§3): vectorized bincount produces same results as loop.
This is a regression test to verify the optimization doesn't change outputs.
"""
import numpy as np
import sys
sys.path.insert(0, '..')


def compute_avg_stats_loop(firm_unit, unit_members, unit_size, active_unit_ids,
                           market_share_current, ranks_step, firm_to_market, firm_to_local_idx):
    """Original loop-based computation."""
    active_sizes = unit_size[active_unit_ids]
    avg_share = np.empty((len(active_unit_ids), 2), dtype=np.float32)
    avg_rank = np.empty((len(active_unit_ids), 2), dtype=np.float32)
    
    for idx, unit_id in enumerate(active_unit_ids):
        n_firms = active_sizes[idx]
        firm_ids = unit_members[unit_id, :n_firms]
        unit_markets = firm_to_market[firm_ids]
        local_idxs = firm_to_local_idx[firm_ids]
        
        avg_share_val = market_share_current[unit_markets, local_idxs].mean()
        avg_rank_val = ranks_step[unit_markets, local_idxs].mean()
        
        avg_share[idx] = [n_firms, avg_share_val]
        avg_rank[idx] = [n_firms, avg_rank_val]
    
    return avg_share, avg_rank


def compute_avg_stats_bincount(firm_unit, unit_size, active_unit_ids,
                               market_share_current, ranks_step, MAX_UNITS):
    """New vectorized bincount computation."""
    active_sizes = unit_size[active_unit_ids]
    
    in_unit_mask = firm_unit != -1
    cids = firm_unit[in_unit_mask]
    cnt = np.bincount(cids, minlength=MAX_UNITS)
    sh_sum = np.bincount(cids, weights=market_share_current.ravel()[in_unit_mask], minlength=MAX_UNITS)
    rk_sum = np.bincount(cids, weights=ranks_step.ravel()[in_unit_mask], minlength=MAX_UNITS)
    
    avg_share = np.column_stack([active_sizes, sh_sum[active_unit_ids] / cnt[active_unit_ids]]).astype(np.float32)
    avg_rank = np.column_stack([active_sizes, rk_sum[active_unit_ids] / cnt[active_unit_ids]]).astype(np.float32)
    
    return avg_share, avg_rank


def test_metrics_equivalence():
    """Test that bincount version produces same results as loop version."""
    np.random.seed(42)
    
    M = 20  # markets
    N = 20  # firms per market
    total_firms = M * N
    MAX_UNITS = total_firms // 2
    
    # Create test data
    firm_to_market = np.repeat(np.arange(M), N)
    firm_to_local_idx = np.tile(np.arange(N), M)
    
    # Random market shares
    market_share_current = np.random.random((M, N)).astype(np.float32)
    market_share_current /= market_share_current.sum(axis=1, keepdims=True)
    
    # Random ranks
    ranks_step = np.random.randint(1, N + 1, (M, N)).astype(np.uint16)
    
    # Create some conglomerates
    firm_unit = np.full(total_firms, -1, dtype=np.int32)
    unit_members = np.full((MAX_UNITS, M), -1, dtype=np.int32)
    unit_size = np.zeros(MAX_UNITS, dtype=np.int16)
    
    # Create 3 conglomerates with different sizes
    # Cong 0: firms 0, 20, 40 (markets 0, 1, 2)
    unit_members[0, :3] = [0, 20, 40]
    unit_size[0] = 3
    for f in [0, 20, 40]:
        firm_unit[f] = 0
    
    # Cong 1: firms 5, 25 (markets 0, 1)
    unit_members[1, :2] = [5, 25]
    unit_size[1] = 2
    for f in [5, 25]:
        firm_unit[f] = 1
    
    # Cong 2: firms 10, 30, 50, 70 (markets 0, 1, 2, 3)
    unit_members[2, :4] = [10, 30, 50, 70]
    unit_size[2] = 4
    for f in [10, 30, 50, 70]:
        firm_unit[f] = 2
    
    active_unit_ids = np.array([0, 1, 2])
    
    # Compute with both methods
    avg_share_loop, avg_rank_loop = compute_avg_stats_loop(
        firm_unit, unit_members, unit_size, active_unit_ids,
        market_share_current, ranks_step, firm_to_market, firm_to_local_idx
    )
    
    avg_share_bincount, avg_rank_bincount = compute_avg_stats_bincount(
        firm_unit, unit_size, active_unit_ids,
        market_share_current, ranks_step, MAX_UNITS
    )
    
    # Compare results
    np.testing.assert_allclose(avg_share_loop, avg_share_bincount, rtol=1e-5,
                               err_msg="avg_share differs between loop and bincount")
    np.testing.assert_allclose(avg_rank_loop, avg_rank_bincount, rtol=1e-5,
                               err_msg="avg_rank differs between loop and bincount")


if __name__ == '__main__':
    print("Testing metrics equivalence (loop vs bincount)...")
    test_metrics_equivalence()
    print("PASS: Vectorized bincount produces identical results to loop")
