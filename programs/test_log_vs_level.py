"""
Test script to verify that log-space and level-space implementations
produce identical results.

This runs a small simulation with both approaches and compares all outputs.
"""

import numpy as np
import sys

# Import both implementations
from collaborative_growth import model as model_level
# We'll need to add the necessary helper functions to TestingLogs.py first
# For now, let's assume we can import it after fixing it


def compare_arrays(arr1, arr2, name, rtol=1e-5, atol=1e-6):
    """
    Compare two arrays and report differences.

    Returns True if arrays match within tolerance, False otherwise.
    """
    if arr1.shape != arr2.shape:
        print(f"❌ {name}: SHAPE MISMATCH - {arr1.shape} vs {arr2.shape}")
        return False

    # Handle NaN/Inf values
    if not np.isfinite(arr1).all() or not np.isfinite(arr2).all():
        finite1 = np.isfinite(arr1)
        finite2 = np.isfinite(arr2)
        if not np.array_equal(finite1, finite2):
            print(f"❌ {name}: Different NaN/Inf patterns")
            return False

    # Compare finite values
    finite_mask = np.isfinite(arr1) & np.isfinite(arr2)
    if finite_mask.any():
        max_abs_diff = np.max(np.abs(arr1[finite_mask] - arr2[finite_mask]))
        max_rel_diff = np.max(np.abs((arr1[finite_mask] - arr2[finite_mask]) /
                                     (np.abs(arr2[finite_mask]) + 1e-10)))

        is_close = np.allclose(arr1[finite_mask], arr2[finite_mask], rtol=rtol, atol=atol)

        if is_close:
            print(f"✅ {name}: MATCH (max_abs_diff={max_abs_diff:.2e}, max_rel_diff={max_rel_diff:.2e})")
            return True
        else:
            print(f"❌ {name}: MISMATCH (max_abs_diff={max_abs_diff:.2e}, max_rel_diff={max_rel_diff:.2e})")

            # Show some example differences
            diff = np.abs(arr1[finite_mask] - arr2[finite_mask])
            worst_indices = np.argsort(diff)[-5:]  # 5 worst
            print(f"   Worst differences:")
            for idx in worst_indices[::-1]:
                print(f"     Index {idx}: level={arr2[finite_mask][idx]:.6e}, log={arr1[finite_mask][idx]:.6e}, diff={diff[idx]:.6e}")
            return False
    else:
        print(f"✅ {name}: All values non-finite and patterns match")
        return True


def test_simple_case():
    """
    Test with a minimal case: few markets, few firms, few steps.
    """
    print("="*80)
    print("SIMPLE TEST CASE")
    print("="*80)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Minimal parameters for fast testing
    params = [
        3,      # markets
        2,      # firms_per_market
        10,     # steps
        0.5,    # share
        6,      # total_firms (3*2)
        0.1,    # merge_thresh
        'avg',  # comparison
        0.8,    # break_thresh
        True,   # proportional
        5,      # lookback
        'linear',  # cost_type
        0.00001,   # c0
        0.00004225,  # c1
        0.001      # c2
    ]

    print(f"Running level-space implementation...")
    np.random.seed(42)  # Reset seed
    results_level = model_level(params)

    print(f"Running log-space implementation...")
    np.random.seed(42)  # Reset seed - should produce identical randomness
    # results_log = model_log(params)  # Will import from TestingLogs once we fix it

    # For now, just show what we'd compare
    print("\nResults structure:")
    print(f"  [0] mean_members: shape {results_level[0].shape}")
    print(f"  [1] quantiles_members: shape {results_level[1].shape}")
    print(f"  [2] num_cong: length {len(results_level[2])}")
    print(f"  [3] avg_shares: length {len(results_level[3])}")
    print(f"  [4] quantiles_shares: shape {results_level[4].shape}")
    print(f"  [5] market_share: shape {results_level[5].shape}")
    print(f"  [6] gini_coefficient: shape {results_level[6].shape}")
    print(f"  [7] ranks: shape {results_level[7].shape}")
    print(f"  [8] avg_ranks: length {len(results_level[8])}")
    print(f"  [9] mergers_per_period: shape {results_level[9].shape}")
    print(f"  [10] hyperparameters: dict with {len(results_level[10])} keys")

    return results_level


def compare_results(results_level, results_log):
    """
    Compare all outputs from both implementations.
    """
    print("\n" + "="*80)
    print("COMPARING RESULTS")
    print("="*80 + "\n")

    all_pass = True

    # Unpack results
    mean_members_l, quantiles_members_l, num_cong_l, avg_shares_l, quantiles_shares_l, \
        market_share_l, gini_l, ranks_l, avg_ranks_l, mergers_l, hyper_l = results_level

    mean_members_log, quantiles_members_log, num_cong_log, avg_shares_log, quantiles_shares_log, \
        market_share_log, gini_log, ranks_log, avg_ranks_log, mergers_log, hyper_log = results_log

    # 1. Mean members
    all_pass &= compare_arrays(mean_members_l, mean_members_log, "mean_members")

    # 2. Quantiles members
    all_pass &= compare_arrays(quantiles_members_l, quantiles_members_log, "quantiles_members")

    # 3. Number of conglomerates
    num_cong_l_arr = np.array(num_cong_l)
    num_cong_log_arr = np.array(num_cong_log)
    all_pass &= compare_arrays(num_cong_l_arr, num_cong_log_arr, "num_cong", atol=0)

    # 4. Average shares (list of arrays)
    print(f"\nComparing avg_shares (list of {len(avg_shares_l)} arrays)...")
    for i, (shares_l, shares_log) in enumerate(zip(avg_shares_l, avg_shares_log)):
        if shares_l.shape[0] > 0 or shares_log.shape[0] > 0:
            result = compare_arrays(shares_l, shares_log, f"  avg_shares[{i}]")
            all_pass &= result
            if not result and i > 3:  # Only show first few
                print(f"  ... (skipping remaining {len(avg_shares_l)-i-1} time steps)")
                break

    # 5. Quantiles shares
    all_pass &= compare_arrays(quantiles_shares_l, quantiles_shares_log, "quantiles_shares")

    # 6. Market share (most important!)
    all_pass &= compare_arrays(market_share_l, market_share_log, "market_share", rtol=1e-4, atol=1e-5)

    # 7. Gini coefficient
    all_pass &= compare_arrays(gini_l, gini_log, "gini_coefficient")

    # 8. Ranks
    all_pass &= compare_arrays(ranks_l, ranks_log, "ranks", atol=0)

    # 9. Average ranks
    print(f"\nComparing avg_ranks (list of {len(avg_ranks_l)} arrays)...")
    for i, (ranks_l_i, ranks_log_i) in enumerate(zip(avg_ranks_l, avg_ranks_log)):
        if ranks_l_i.shape[0] > 0 or ranks_log_i.shape[0] > 0:
            result = compare_arrays(ranks_l_i, ranks_log_i, f"  avg_ranks[{i}]")
            all_pass &= result
            if not result and i > 3:
                print(f"  ... (skipping remaining {len(avg_ranks_l)-i-1} time steps)")
                break

    # 10. Mergers per period
    all_pass &= compare_arrays(mergers_l, mergers_log, "mergers_per_period", atol=0)

    # 11. Hyperparameters (should be identical)
    print(f"\nComparing hyperparameters...")
    hyper_match = hyper_l == hyper_log
    if hyper_match:
        print(f"✅ hyperparameters: MATCH")
    else:
        print(f"❌ hyperparameters: MISMATCH")
        for key in hyper_l.keys():
            if hyper_l[key] != hyper_log.get(key):
                print(f"   {key}: {hyper_l[key]} vs {hyper_log.get(key)}")
        all_pass = False

    print("\n" + "="*80)
    if all_pass:
        print("✅ ALL TESTS PASSED - Implementations are equivalent!")
    else:
        print("❌ SOME TESTS FAILED - Implementations differ!")
    print("="*80 + "\n")

    return all_pass


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LOG-SPACE vs LEVEL-SPACE EQUIVALENCE TEST")
    print("="*80 + "\n")

    # Run simple test
    results_level = test_simple_case()

    print("\n⚠️  NOTE: Log-space implementation import is disabled until TestingLogs.py")
    print("   is complete with all helper functions (management_cost_function, exit_, etc.)")
    print("\nTo enable full testing:")
    print("  1. Add missing helper functions to TestingLogs.py")
    print("  2. Uncomment the log-space model import and call")
    print("  3. Run: python test_log_vs_level.py")

    # Once TestingLogs.py is complete, uncomment:
    # from TestingLogs import model as model_log
    # results_log = model_log(params)
    # all_pass = compare_results(results_level, results_log)
    # sys.exit(0 if all_pass else 1)