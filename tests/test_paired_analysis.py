#!/usr/bin/env python3
"""
Test paired analysis functionality (§5).

- Verifies experiment matching by experiment_id
- Tests paired difference computation
- Tests t-test statistics
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from paired_analysis import (
    match_experiments,
    compute_paired_differences,
    paired_t_test
)


def test_match_experiments_perfect():
    """
    All experiments should match when IDs are the same.
    """
    control = [
        {'experiment_id': 0, 'gini_avg': [0.3, 0.31, 0.32]},
        {'experiment_id': 1, 'gini_avg': [0.35, 0.36, 0.37]},
        {'experiment_id': 2, 'gini_avg': [0.28, 0.29, 0.30]},
    ]
    treatment = [
        {'experiment_id': 0, 'gini_avg': [0.4, 0.41, 0.42]},
        {'experiment_id': 1, 'gini_avg': [0.45, 0.46, 0.47]},
        {'experiment_id': 2, 'gini_avg': [0.38, 0.39, 0.40]},
    ]

    matched, unmatched_ctrl, unmatched_treat = match_experiments(control, treatment)

    assert len(matched) == 3, f"Expected 3 matched pairs, got {len(matched)}"
    assert unmatched_ctrl == 0, "Should have no unmatched control"
    assert unmatched_treat == 0, "Should have no unmatched treatment"

    # Check ordering (should be sorted by experiment_id)
    for i, (ctrl, treat) in enumerate(matched):
        assert ctrl['experiment_id'] == i
        assert treat['experiment_id'] == i

    print("Perfect matching test passed")


def test_match_experiments_partial():
    """
    Only matching IDs should be paired.
    """
    control = [
        {'experiment_id': 0, 'gini_avg': [0.3]},
        {'experiment_id': 2, 'gini_avg': [0.28]},  # Missing ID 1
    ]
    treatment = [
        {'experiment_id': 1, 'gini_avg': [0.45]},  # Missing ID 0
        {'experiment_id': 2, 'gini_avg': [0.38]},
    ]

    matched, unmatched_ctrl, unmatched_treat = match_experiments(control, treatment)

    assert len(matched) == 1, f"Expected 1 matched pair (ID 2), got {len(matched)}"
    assert matched[0][0]['experiment_id'] == 2
    assert matched[0][1]['experiment_id'] == 2
    assert unmatched_ctrl == 1  # ID 0 unmatched
    assert unmatched_treat == 1  # ID 1 unmatched

    print("Partial matching test passed")


def test_compute_paired_differences():
    """
    Differences should be treatment - control.
    """
    matched_pairs = [
        (
            {'gini_avg': np.array([0.3, 0.31, 0.32])},
            {'gini_avg': np.array([0.4, 0.41, 0.42])}
        ),
        (
            {'gini_avg': np.array([0.35, 0.36, 0.37])},
            {'gini_avg': np.array([0.45, 0.46, 0.47])}
        ),
    ]

    diffs = compute_paired_differences(matched_pairs, 'gini_avg')

    assert diffs is not None
    assert diffs.shape == (2, 3), f"Expected shape (2, 3), got {diffs.shape}"

    # Check first pair: 0.4-0.3=0.1, 0.41-0.31=0.1, 0.42-0.32=0.1
    np.testing.assert_array_almost_equal(diffs[0], [0.1, 0.1, 0.1])

    # Check second pair: 0.45-0.35=0.1, etc.
    np.testing.assert_array_almost_equal(diffs[1], [0.1, 0.1, 0.1])

    print("Paired differences test passed")


def test_paired_t_test_known_values():
    """
    Test t-test with known values.
    """
    # Differences with known mean and std
    # Mean = 0.1, std = 0.02, n = 4, SE = 0.02/sqrt(4) = 0.01, t = 0.1/0.01 = 10
    differences = np.array([0.08, 0.10, 0.12, 0.10])

    mean_diff, se_diff, t_stat, n = paired_t_test(differences)

    assert n == 4
    np.testing.assert_almost_equal(mean_diff, 0.10)

    # std with ddof=1: sqrt(sum((x-mean)^2) / (n-1))
    expected_std = np.std(differences, ddof=1)
    expected_se = expected_std / 2  # sqrt(4)

    np.testing.assert_almost_equal(se_diff, expected_se, decimal=6)
    np.testing.assert_almost_equal(t_stat, mean_diff / expected_se, decimal=6)

    print("Paired t-test test passed")


def test_paired_t_test_significance():
    """
    Large effect should be significant, small effect not.
    """
    # Large effect (clearly positive)
    large_diffs = np.array([0.10, 0.11, 0.09, 0.10, 0.12, 0.08, 0.11, 0.10])
    mean_diff, se_diff, t_stat, n = paired_t_test(large_diffs)
    assert abs(t_stat) > 1.96, f"Large effect should be significant, t={t_stat}"

    # Zero effect (noise around zero)
    zero_diffs = np.array([0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.00, 0.00])
    mean_diff, se_diff, t_stat, n = paired_t_test(zero_diffs)
    assert abs(t_stat) < 1.96, f"Zero effect should not be significant, t={t_stat}"

    print("Significance test passed")


if __name__ == '__main__':
    print("Testing perfect matching...")
    test_match_experiments_perfect()
    print("PASS\n")

    print("Testing partial matching...")
    test_match_experiments_partial()
    print("PASS\n")

    print("Testing paired differences...")
    test_compute_paired_differences()
    print("PASS\n")

    print("Testing paired t-test with known values...")
    test_paired_t_test_known_values()
    print("PASS\n")

    print("Testing significance detection...")
    test_paired_t_test_significance()
    print("PASS\n")

    print("All paired analysis tests passed!")
