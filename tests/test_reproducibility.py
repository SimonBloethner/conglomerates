"""
Test reproducibility: two runs with same seed should produce identical results.
§1a/§1b tests.
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from collaborative_growth import model, seed_numba


def arrays_equal(a1, a2):
    """Compare two values, handling numpy arrays with NaN and nested structures."""
    if isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
        # Handle NaN comparisons: np.nan == np.nan is False, so use array_equal with equal_nan
        if a1.dtype.kind == 'f' and a2.dtype.kind == 'f':
            return np.allclose(a1, a2, equal_nan=True) or np.array_equal(a1, a2)
        return np.array_equal(a1, a2)
    elif isinstance(a1, dict) and isinstance(a2, dict):
        if set(a1.keys()) != set(a2.keys()):
            return False
        return all(arrays_equal(a1[k], a2[k]) for k in a1)
    elif isinstance(a1, list) and isinstance(a2, list):
        if len(a1) != len(a2):
            return False
        return all(arrays_equal(x, y) for x, y in zip(a1, a2))
    elif isinstance(a1, tuple) and isinstance(a2, tuple):
        if len(a1) != len(a2):
            return False
        return all(arrays_equal(x, y) for x, y in zip(a1, a2))
    elif isinstance(a1, float) and isinstance(a2, float):
        # Handle NaN float comparison
        if np.isnan(a1) and np.isnan(a2):
            return True
        return a1 == a2
    else:
        return a1 == a2


def test_reproducibility_same_seed():
    """Run model twice with same seed; assert all arrays exactly equal."""
    params = [30, 30, 200, 0.2, 900, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # First run
    result1 = model(params, seed=42, market_corr='identity')

    # Second run with same seed
    result2 = model(params, seed=42, market_corr='identity')

    # Compare all returned elements
    for i, (arr1, arr2) in enumerate(zip(result1, result2)):
        assert arrays_equal(arr1, arr2), f"Element {i} differs between runs"


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
