#!/usr/bin/env python3
"""
Test Phase A identity: current code with default flags must produce
identical results to Phase A reference (commit 56ee13f).

This test must pass at every commit that modifies collaborative_growth.py
or parallel_counterfactuals.py.

Note: ranks array has been removed from current output to save memory.
The test compares all other arrays between Phase A reference and current code.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

# Import Phase A reference (vendored from commit 56ee13f)
from tests import _phase_a_reference as phase_a

# Import current code
import collaborative_growth as current


def test_phase_a_identity():
    """
    Run model at M=N=20, T=300, α=0.2, seed=1, default flags on both
    Phase A reference and current code. Assert every returned array
    is exactly equal (except ranks which has been removed from current output).
    """
    # Phase A parameters: M=20, N=20, T=300, α=0.2
    M = 20
    N = 20
    T = 300
    alpha = 0.2
    total_firms = M * N

    # Standard Phase A params list (14 elements)
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

    # Run Phase A reference
    print("Running Phase A reference...")
    result_ref = phase_a.model(params, seed=seed, market_corr='identity')

    # Run current code with Phase A defaults
    print("Running current code with Phase A defaults...")
    result_cur = current.model(params, seed=seed, market_corr='identity')

    # Phase A reference structure (12 elements):
    # 0: mean_members
    # 1: quantiles_members
    # 2: num_cong
    # 3: avg_shares (list of arrays)
    # 4: quantiles_shares
    # 5: gini_coefficient
    # 6: ranks
    # 7: avg_ranks (list of arrays)
    # 8: mergers_per_period
    # 9: proposals_per_period
    # 10: exits_per_period
    # 11: hyperparameters (dict)

    # Current structure (11 elements - ranks removed):
    # 0: mean_members
    # 1: quantiles_members
    # 2: num_cong
    # 3: avg_shares (list of arrays)
    # 4: quantiles_shares
    # 5: gini_coefficient
    # 6: avg_ranks (list of arrays)  <- was index 7
    # 7: mergers_per_period           <- was index 8
    # 8: proposals_per_period         <- was index 9
    # 9: exits_per_period             <- was index 10
    # 10: hyperparameters (dict)

    # Map: (name, ref_index, cur_index)
    comparisons = [
        ('mean_members', 0, 0),
        ('quantiles_members', 1, 1),
        ('num_cong', 2, 2),
        ('avg_shares', 3, 3),
        ('quantiles_shares', 4, 4),
        ('gini_coefficient', 5, 5),
        # ranks (index 6 in ref) is skipped - removed from current
        ('avg_ranks', 7, 6),
        ('mergers_per_period', 8, 7),
        ('proposals_per_period', 9, 8),
        ('exits_per_period', 10, 9),
    ]

    # Compare arrays
    all_match = True
    for name, ref_idx, cur_idx in comparisons:
        ref_val = result_ref[ref_idx]
        cur_val = result_cur[cur_idx]

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

    assert all_match, "Phase A identity test failed: some arrays differ"
    print("\nPhase A identity test PASSED")


if __name__ == '__main__':
    test_phase_a_identity()
