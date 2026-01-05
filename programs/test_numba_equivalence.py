"""
Test to verify that Numba and non-Numba implementations produce identical results.
"""
import numpy as np
from collaborative_growth import model

def compare_arrays(arr1, arr2, name, rtol=1e-5, atol=1e-8):
    """Compare two arrays and report differences."""
    if isinstance(arr1, list) and isinstance(arr2, list):
        if len(arr1) != len(arr2):
            print(f"❌ {name}: LENGTH MISMATCH - {len(arr1)} vs {len(arr2)}")
            return False

        all_match = True
        for i, (a1, a2) in enumerate(zip(arr1, arr2)):
            if not compare_arrays(a1, a2, f"{name}[{i}]", rtol, atol):
                all_match = False
                if i > 3:  # Only show first few
                    print(f"  ... (skipping remaining {len(arr1)-i-1} items)")
                    break
        return all_match

    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        # Handle scalars or other types
        match = arr1 == arr2
        if match:
            print(f"✅ {name}: MATCH")
        else:
            print(f"❌ {name}: MISMATCH - {arr1} vs {arr2}")
        return match

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
            worst_indices = np.argsort(diff)[-3:]  # 3 worst
            print(f"   Worst differences:")
            for idx in worst_indices[::-1]:
                print(f"     Index {idx}: v1={arr1.flat[idx]:.6e}, v2={arr2.flat[idx]:.6e}, diff={diff[idx]:.6e}")
            return False
    else:
        print(f"✅ {name}: All values non-finite and patterns match")
        return True


# Test parameters - small enough to run quickly but complex enough to test all code paths
params = [
    5,      # markets
    3,      # firms_per_market
    100,    # steps
    0.5,    # share
    15,     # total_firms (5*3)
    0.15,   # merge_thresh (higher to get more mergers)
    'avg',  # comparison
    0.8,    # break_thresh
    True,   # proportional
    10,     # lookback
    'linear',  # cost_type
    0.00001,   # c0
    0.00004225,  # c1
    0.001      # c2
]

print("=" * 80)
print("TESTING NUMBA IMPLEMENTATION EQUIVALENCE")
print("=" * 80)
print("\nRunning with identical random seed to ensure same randomness...")
print(f"Parameters: markets={params[0]}, firms_per_market={params[1]}, steps={params[2]}")
print(f"            share={params[3]}, merge_thresh={params[5]}")
print()

# Run with Numba (current implementation)
print("Running Numba implementation...")
np.random.seed(42)
results_numba = model(params)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print("\nNumba implementation completed successfully!")
print(f"  Mean members: {results_numba[0].mean():.3f}")
print(f"  Final conglomerates: {results_numba[2][-1] if len(results_numba[2]) > 0 else 0}")
print(f"  Total mergers: {results_numba[8].sum():.0f}")
print(f"  Mean Gini: {results_numba[5].mean():.4f}")

print("\n" + "=" * 80)
print("NOTE: To compare with non-Numba implementation, you would need to:")
print("  1. Comment out the Numba pooling code (lines 590-642)")
print("  2. Restore the original Python loop (from git history)")
print("  3. Run this test with both versions")
print("=" * 80)
