#!/usr/bin/env python3
"""
Tests for C13: Floor calibration.

Tests verify:
1. Hill exponent is monotone non-decreasing in c on the N=50 grid (median values)
   Catches: wrong share normalization, or Hill computed on sizes instead of shares
2. At N=200 and c=0.3, exponent is within 20% of 1/(1-0.3) = 1.4286
   Catches: broken floor. Note: floor_share = c/N, so at N=200 with c=0.0566 the
   floor share is only 0.03% and has no meaningful bite. We use c=0.3 where floor
   share = 0.15% is still small but the absolute floor is high enough to affect
   the distribution.
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


def test_hill_monotone_nondecreasing_n50():
    """
    Hill exponent is monotone non-decreasing in c on the N=50 grid.

    Catches: wrong share normalization, or Hill computed on sizes instead of shares.
    The floor reflection raises the floor and should increase the Hill exponent.
    """
    df = load_calibration_csv()

    # Filter to N=50
    n50 = df[df['N'] == 50].sort_values('c')

    c_values = n50['c'].values
    hill_values = n50['hill_median'].values

    print(f"N=50 grid:")
    for c, h in zip(c_values, hill_values):
        print(f"  c={c:.4f}: Hill={h:.4f}")

    # Check monotonicity
    for i in range(1, len(hill_values)):
        assert hill_values[i] >= hill_values[i - 1] - 1e-6, \
            f"Hill not monotone: c={c_values[i-1]:.4f} -> c={c_values[i]:.4f}: " \
            f"Hill {hill_values[i-1]:.4f} -> {hill_values[i]:.4f}"

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


if __name__ == '__main__':
    print("=" * 60)
    print("C13: Floor Calibration Tests")
    print("=" * 60)

    tests = [
        ("Hill monotone in c (N=50)", test_hill_monotone_nondecreasing_n50),
        ("Hill near asymptotic (N=200, c=0.3)", test_hill_near_asymptotic_n200),
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
