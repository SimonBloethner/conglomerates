#!/usr/bin/env python3
"""
Tests for C13: Floor calibration.

Tests verify:
1. Hill exponent is monotone non-decreasing in c on the N=50 grid (median values), and (C32) on the
   N=25 and N=100 grids, between grid points at least 0.02 apart in c
   Catches: wrong share normalization, or Hill computed on sizes instead of shares
2. At N=200 and c=0.3, exponent is within 20% of 1/(1-0.3) = 1.4286
   Catches: broken floor. Note: floor_share = c/N, so at N=200 with c=0.0566 the
   floor share is only 0.03% and has no meaningful bite. We use c=0.3 where floor
   share = 0.15% is still small but the absolute floor is high enough to affect
   the distribution.
3. (C32) c*(N) for N = 25, 50, 100 is bracketed by two adjacent grid points (interpolation, not extrapolation)
"""
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_calibration_csv(filepath=None):
    """Load floor calibration CSV."""
    if filepath is None:
        # Default to analytics/floor_calibration.csv relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(project_root, 'analytics', 'floor_calibration.csv')
    return pd.read_csv(filepath)


MIN_DC = 0.02  # smallest c spacing between grid points that the monotonicity tests compare


def assert_hill_monotone(df, N):
    """Hill median non-decreasing (tolerance 1e-6) between every pair of N's grid points at least MIN_DC apart.

    Why not every adjacent pair: the Hill-c slope is about 3, so 0.02 in c moves Hill by about 0.06, against a
    replication SE of the median of about 0.015 at 20 reps (0.03 at 5). Two points 0.0066 apart (0.05, 0.0566)
    are inside the noise at any affordable rep count; a strict adjacent-pair test there measures seeds, not the
    model."""
    grid = df[df['N'] == N].sort_values('c')
    c_values = grid['c'].values
    hill_values = grid['hill_median'].values
    n_pairs = 0
    for i in range(len(c_values)):
        for j in range(i + 1, len(c_values)):
            if c_values[j] - c_values[i] < MIN_DC - 1e-12:
                continue
            n_pairs += 1
            assert hill_values[j] >= hill_values[i] - 1e-6, \
                f"N={N} Hill not monotone: c={c_values[i]:.4f} -> c={c_values[j]:.4f}: " \
                f"Hill {hill_values[i]:.4f} -> {hill_values[j]:.4f}"
    assert n_pairs > 0, f"N={N}: no grid pairs tested"
    return c_values, hill_values


def test_hill_monotone_nondecreasing_n50():
    """
    Hill exponent is monotone non-decreasing in c on the N=50 grid, between grid points at least 0.02 apart
    (rule and reason in assert_hill_monotone).

    Catches: wrong share normalization, or Hill computed on sizes instead of shares.
    The floor reflection raises the floor and should increase the Hill exponent.
    """
    df = load_calibration_csv()
    c_values, hill_values = assert_hill_monotone(df, 50)

    print(f"N=50 grid:")
    for c, h in zip(c_values, hill_values):
        print(f"  c={c:.4f}: Hill={h:.4f}")
    print("PASS: Hill exponent monotone non-decreasing in c (N=50)")


def test_hill_near_asymptotic_n200():
    """
    At N=200 and c=0.3, exponent is within 20% of 1/(1-0.3) = 1.4286.

    Catches: broken floor. Note: floor_share = c/N, so with c=0.0566 at N=200
    the floor share is only 0.03% and has no meaningful bite. We use c=0.3
    where the floor share is 0.15% -- still small but the absolute floor is
    high enough to meaningfully affect the tail distribution.
    """
    df = load_calibration_csv()

    c_test = 0.3
    # Filter to N=200, c=0.3
    n200_c03 = df[(df['N'] == 200) & (np.abs(df['c'] - c_test) < 0.001)]

    if len(n200_c03) == 0:
        raise AssertionError(f"No N=200, c={c_test} results found in calibration CSV")

    hill = n200_c03['hill_median'].iloc[0]
    theoretical = 1.0 / (1.0 - c_test)  # = 1.4286

    print(f"N=200, c={c_test}:")
    print(f"  Hill = {hill:.4f}")
    print(f"  Theoretical 1/(1-c) = {theoretical:.4f}")
    print(f"  Ratio = {hill / theoretical:.4f}")
    print(f"  Floor share at N=200: {c_test/200:.4f} = {100*c_test/200:.2f}%")

    # Within 20% means 0.8 * theoretical < hill < 1.2 * theoretical
    lower = 0.8 * theoretical
    upper = 1.2 * theoretical

    assert lower < hill < upper, \
        f"Hill ({hill:.4f}) not within 20% of theoretical ({theoretical:.4f}): " \
        f"expected [{lower:.4f}, {upper:.4f}]"

    print("PASS: Hill at N=200, c=0.3 within 20% of theoretical")


def test_hill_monotone_n25_n100():
    """
    C32: Hill exponent is monotone non-decreasing in c on the N=25 (20 reps) and N=100 (5 reps) grids, between
    grid points at least 0.02 apart, the same rule as the N=50 test (reason in assert_hill_monotone).

    Catches: wrong share normalization, or Hill computed on sizes instead of shares, at the new N.
    """
    df = load_calibration_csv()

    for N in (25, 100):
        assert len(df[df['N'] == N]) == {25: 9, 100: 6}[N], f"N={N}: wrong number of grid points"
        assert_hill_monotone(df, N)


def test_c_star_bracketed():
    """
    C32: for each N in (25, 50, 100) the target Hill 1.06 lies between the Hill values of two adjacent grid
    points, so c*(N) is an interpolation, not an extrapolation.

    Catches: a c* read off outside the grid (target above the largest or below the smallest Hill).
    """
    df = load_calibration_csv()
    target = 1.06

    for N in (25, 50, 100):
        grid = df[df['N'] == N].sort_values('c')
        h = grid['hill_median'].values
        bracketed = [(grid['c'].values[i], grid['c'].values[i + 1]) for i in range(len(h) - 1)
                     if min(h[i], h[i + 1]) <= target <= max(h[i], h[i + 1])]
        assert len(bracketed) >= 1, f"N={N}: target {target} not bracketed by Hill values {list(h)}"


if __name__ == '__main__':
    print("=" * 60)
    print("C13: Floor Calibration Tests")
    print("=" * 60)

    tests = [
        ("Hill monotone in c (N=50)", test_hill_monotone_nondecreasing_n50),
        ("Hill near asymptotic (N=200, c=0.3)", test_hill_near_asymptotic_n200),
        ("Hill monotone in c (N=25, N=100)", test_hill_monotone_n25_n100),
        ("c* bracketed by grid (N=25, 50, 100)", test_c_star_bracketed),
    ]

    failed = 0
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        print("-" * 40)
        try:
            test_fn()
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
            print("Run analytics/calibrate_floor.py first to generate results.")
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("All tests PASSED")
    else:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
    print("=" * 60)
