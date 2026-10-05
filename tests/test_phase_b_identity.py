#!/usr/bin/env python3
"""
Test Phase B identity: current code with default flags must produce
identical results to Phase B reference (commit 1650ea9).

This test must pass at every commit that modifies collaborative_growth.py
or parallel_counterfactuals.py.

Test catches: any change to default behaviour in later cards.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

# Import Phase B reference (vendored from commit 1650ea9)
import _phase_b_reference as phase_b

# Import current code
import collaborative_growth as current


def test_phase_b_identity():
    """
    Run model at M=N=20, T=300, α=0.2, seed=1, default flags on both
    Phase B reference and current code. Assert every returned array
    is exactly equal.
    """
    # Parameters: M=20, N=20, T=300, α=0.2
    M = 20
    N = 20
    T = 300
    alpha = 0.2
    total_firms = M * N

    # Standard params list (14 elements)
    params = [
        M,                # markets
        N,                # firms_per_market
        T,                # steps
        alpha,            # share
        total_firms,      # total_firms
        0.05,             # merge_thresh
        4,                # comparison
        0.85,             # break_thresh
        False,            # proportional
        50,               # lookback
        'power_law',      # cost_type
        None,             # c0 (use default)
        None,             # c1 (use default)
        None,             # c2 (use default)
    ]

    seed = 1

    # Run Phase B reference
    print("Running Phase B reference...")
    result_ref = phase_b.model(params, seed=seed, market_corr='identity')

    # Run current code
    print("Running current code...")
    result_cur = current.model(params, seed=seed, market_corr='identity')

    # Phase B structure (13 elements):
    # 0: mean_members
    # 1: quantiles_members
    # 2: num_cong
    # 3: avg_shares (list of arrays)
    # 4: quantiles_shares
    # 5: gini_coefficient
    # 6: avg_ranks (list of arrays)
    # 7: mergers_per_period
    # 8: proposals_per_period
    # 9: exits_per_period
    # 10: rank_range
    # 11: rank_std
    # 12: hyperparameters (dict)

    # Verify output lengths
    assert len(result_ref) == 13, f"Phase B reference should have 13 elements, got {len(result_ref)}"
    assert len(result_cur) == 13, f"Current code should have 13 elements, got {len(result_cur)}"

    # Array comparisons (name, index)
    comparisons = [
        ('mean_members', 0),
        ('quantiles_members', 1),
        ('num_cong', 2),
        ('avg_shares', 3),
        ('quantiles_shares', 4),
        ('gini_coefficient', 5),
        ('avg_ranks', 6),
        ('mergers_per_period', 7),
        ('proposals_per_period', 8),
        ('exits_per_period', 9),
        ('rank_range', 10),
        ('rank_std', 11),
    ]

    # Compare arrays
    all_match = True
    for name, idx in comparisons:
        ref_val = result_ref[idx]
        cur_val = result_cur[idx]

        if isinstance(ref_val, list):
            # avg_shares and avg_ranks are lists of arrays
            if len(ref_val) != len(cur_val):
                print(f"FAIL: {name} length mismatch: {len(ref_val)} vs {len(cur_val)}")
                all_match = False
                continue
            for j, (r, c) in enumerate(zip(ref_val, cur_val)):
                if not np.array_equal(r, c):
                    print(f"FAIL: {name}[{j}] differs")
                    all_match = False
                    break
            else:
                print(f"OK: {name}")
        else:
            # Regular numpy array
            if not np.array_equal(ref_val, cur_val):
                print(f"FAIL: {name} differs")
                if hasattr(ref_val, 'shape') and hasattr(cur_val, 'shape'):
                    print(f"  Shape: {ref_val.shape} vs {cur_val.shape}")
                if hasattr(ref_val, 'dtype') and hasattr(cur_val, 'dtype'):
                    print(f"  Dtype: {ref_val.dtype} vs {cur_val.dtype}")
                # Show first difference
                if hasattr(ref_val, 'ravel') and hasattr(cur_val, 'ravel'):
                    r_flat = ref_val.ravel()
                    c_flat = cur_val.ravel()
                    for k in range(min(len(r_flat), len(c_flat))):
                        if r_flat[k] != c_flat[k]:
                            print(f"  First diff at index {k}: {r_flat[k]} vs {c_flat[k]}")
                            break
                all_match = False
            else:
                print(f"OK: {name}")

    assert all_match, "Phase B identity test failed: some arrays differ"
    print("\nPhase B identity test PASSED")


if __name__ == '__main__':
    test_phase_b_identity()
