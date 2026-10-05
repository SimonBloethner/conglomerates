"""
Test reproducibility: two runs with same seed should produce identical results.
§1a/§1b tests.
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from collaborative_growth import model, seed_numba


def test_reproducibility_same_seed():
    """Run model twice with same seed; assert all arrays exactly equal."""
    params = [30, 30, 200, 0.2, 900, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]
    
    # First run
    result1 = model(params, seed=42, market_corr='identity')
    
    # Second run with same seed
    result2 = model(params, seed=42, market_corr='identity')
    
    # Compare all returned arrays
    for i, (arr1, arr2) in enumerate(zip(result1, result2)):
        if isinstance(arr1, np.ndarray):
            assert np.array_equal(arr1, arr2), f"Array {i} differs between runs"
        elif isinstance(arr1, list):
            for j, (a1, a2) in enumerate(zip(arr1, arr2)):
                if isinstance(a1, np.ndarray):
                    assert np.array_equal(a1, a2), f"List {i} element {j} differs"
        elif isinstance(arr1, dict):
            for k in arr1:
                v1, v2 = arr1[k], arr2[k]
                if isinstance(v1, np.ndarray):
                    assert np.array_equal(v1, v2), f"Dict {i} key {k} differs"
                else:
                    assert v1 == v2, f"Dict {i} key {k} differs"


def test_reproducibility_different_seed():
    """Run model with different seeds; assert results differ."""
    params = [30, 30, 200, 0.2, 900, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]
    
    # Run with seed 42
    result1 = model(params, seed=42, market_corr='identity')
    
    # Run with seed 123
    result2 = model(params, seed=123, market_corr='identity')
    
    # At least one array should differ
    any_different = False
    for i, (arr1, arr2) in enumerate(zip(result1, result2)):
        if isinstance(arr1, np.ndarray) and arr1.size > 0:
            if not np.array_equal(arr1, arr2):
                any_different = True
                break
    
    assert any_different, "Results should differ with different seeds"


if __name__ == '__main__':
    print("Testing reproducibility with same seed...")
    test_reproducibility_same_seed()
    print("PASS: Same seed produces identical results")
    
    print("Testing reproducibility with different seed...")
    test_reproducibility_different_seed()
    print("PASS: Different seeds produce different results")
