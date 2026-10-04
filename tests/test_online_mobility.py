#!/usr/bin/env python3
"""
Test online rank statistics.

Validates:
1. Model returns correct output structure (13 elements with rank_range, rank_std)
2. Rank statistics have reasonable values
3. Results are reproducible with the same seed
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from collaborative_growth import model


def test_rank_stats_output_structure():
    """
    Verify model returns 13 elements with rank_range and rank_std at correct indices.
    """
    params = [10, 10, 100, 0.0, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result = model(params, seed=123, market_corr='identity')

    assert len(result) == 13, f"Expected 13 elements, got {len(result)}"

    # rank_range at index 10
    rank_range = result[10]
    assert isinstance(rank_range, np.ndarray), "rank_range should be numpy array"
    assert rank_range.shape == (10, 10), f"rank_range shape should be (M, N), got {rank_range.shape}"
    assert rank_range.dtype == np.float32, f"rank_range dtype should be float32, got {rank_range.dtype}"

    # rank_std at index 11
    rank_std = result[11]
    assert isinstance(rank_std, np.ndarray), "rank_std should be numpy array"
    assert rank_std.shape == (10, 10), f"rank_std shape should be (M, N), got {rank_std.shape}"
    assert rank_std.dtype == np.float32, f"rank_std dtype should be float32, got {rank_std.dtype}"

    # hyperparameters at index 12
    hyperparameters = result[12]
    assert isinstance(hyperparameters, dict), "hyperparameters should be dict"

    print("Output structure verified: 13 elements with rank_range, rank_std, hyperparameters")


def test_rank_stats_reasonable_values():
    """
    Verify rank statistics have reasonable values.
    """
    M, N, T = 15, 15, 200
    params = [M, N, T, 0.0, M * N, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result = model(params, seed=42, market_corr='identity')

    rank_range = result[10]
    rank_std = result[11]

    # rank_range should be in [0, N-1] (max possible range)
    assert np.all(rank_range >= 0), "rank_range should be non-negative"
    assert np.all(rank_range <= N - 1), f"rank_range should be <= {N-1}"

    # rank_std should be in [0, (N-1)/2] approximately (max std for uniform on 1..N)
    assert np.all(rank_std >= 0), "rank_std should be non-negative"
    max_possible_std = (N - 1) / 2  # conservative upper bound
    assert np.all(rank_std <= max_possible_std), f"rank_std should be <= {max_possible_std}"

    # With volatile growth, most firms should have rank_range > 0
    assert np.mean(rank_range > 0) > 0.5, "Most firms should have some rank mobility"

    print(f"rank_range: mean={np.mean(rank_range):.2f}, max={np.max(rank_range):.0f}")
    print(f"rank_std: mean={np.mean(rank_std):.2f}, max={np.max(rank_std):.2f}")
    print("Rank statistics have reasonable values")


def test_rank_stats_reproducibility():
    """
    Verify rank statistics are reproducible with the same seed.
    """
    M, N, T = 20, 20, 300
    params = [M, N, T, 0.1, M * N, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Run twice with same seed
    result1 = model(params, seed=12345, market_corr='identity')
    result2 = model(params, seed=12345, market_corr='identity')

    rank_range1 = result1[10]
    rank_range2 = result2[10]
    rank_std1 = result1[11]
    rank_std2 = result2[11]

    # Should be bit-identical
    assert np.array_equal(rank_range1, rank_range2), "rank_range differs between runs"
    assert np.array_equal(rank_std1, rank_std2), "rank_std differs between runs"

    print("Rank statistics reproducibility verified")


def test_rank_stats_vary_with_seed():
    """
    Verify rank statistics differ with different seeds.
    """
    M, N, T = 15, 15, 200
    params = [M, N, T, 0.0, M * N, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result1 = model(params, seed=111, market_corr='identity')
    result2 = model(params, seed=222, market_corr='identity')

    rank_range1 = result1[10]
    rank_range2 = result2[10]

    # Should differ
    assert not np.array_equal(rank_range1, rank_range2), \
        "rank_range should differ with different seeds"

    print("Rank statistics vary with seed")


if __name__ == '__main__':
    print("Testing output structure...")
    test_rank_stats_output_structure()
    print("PASS\n")

    print("Testing reasonable values...")
    test_rank_stats_reasonable_values()
    print("PASS\n")

    print("Testing reproducibility...")
    test_rank_stats_reproducibility()
    print("PASS\n")

    print("Testing seed variation...")
    test_rank_stats_vary_with_seed()
    print("PASS\n")

    print("All online mobility tests passed!")
