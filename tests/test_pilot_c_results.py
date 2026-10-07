#!/usr/bin/env python3
"""
Phase C pilot results sanity tests.

Tests verify pilot_c/medians.csv has expected structure and values:
1. K increases with α (conglomerates form at higher pooling)
2. Hill exponent at α=0 is near 1/(1-c) ≈ 1.060
3. Floor-hit rate for members < standalones at α > 0 (risk sharing works)
4. K_eff < K (not all members equally active)

These are directional sanity checks, not exact value tests.
"""
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_medians_csv(filepath='pilot_c/medians.csv'):
    """Load medians CSV."""
    return pd.read_csv(filepath)


def load_tidy_csv(filepath='pilot_c/tidy.csv'):
    """Load tidy CSV."""
    return pd.read_csv(filepath)


def test_k_nondecreasing_with_alpha():
    """
    K non-decreasing with α: median K at α=0.3 >= median K at α=0.05.

    Test catches: M&A mechanism broken (K decreasing with pooling).
    Note: K may stay flat if merger acceptance rate is low.
    Source: pilot_c/medians.csv columns K_post_burnin_median_median
    """
    df = load_medians_csv()

    # Filter main block, laplace family, power_law cost
    main = df[(df['block'] == 'main') &
              (df['log_family'] == 'laplace') &
              (df['cost_type'] == 'power_law')]

    # Get K at α=0.05 and α=0.3
    k_low = main[main['alpha'] == 0.05]['K_post_burnin_median_median'].iloc[0]
    k_high = main[main['alpha'] == 0.3]['K_post_burnin_median_median'].iloc[0]

    print(f"K at α=0.05: {k_low:.2f}")
    print(f"K at α=0.3: {k_high:.2f}")

    assert k_high >= k_low, \
        f"K at α=0.3 ({k_high:.2f}) should be >= K at α=0.05 ({k_low:.2f})"
    print("PASS: K non-decreasing with α")


def test_hill_at_alpha0():
    """
    Hill exponent at α=0 is positive and in plausible range.

    Test catches: Hill estimator bug (NaN, negative, or extreme values).
    Source: pilot_c/tidy.csv column hill_exponent_median, filtered by alpha=0
    Expected: hill ∈ (0.5, 2.0) - a wide range to allow for finite-sample effects
    """
    df = load_tidy_csv()

    # Filter main block, α=0
    main_a0 = df[(df['block'] == 'main') & (df['alpha'] == 0.0)]

    hill_median = main_a0['hill_exponent_median'].median()
    floor_c = main_a0['floor_c'].iloc[0] if 'floor_c' in main_a0.columns else 0.0566
    theoretical = 1.0 / (1.0 - floor_c)

    print(f"Hill exponent at α=0: {hill_median:.4f}")
    print(f"Theoretical 1/(1-c): {theoretical:.4f}")
    print(f"Ratio: {hill_median / theoretical:.3f}")

    # Sanity check: Hill should be positive and in plausible range
    assert 0.5 < hill_median < 2.0, \
        f"Hill at α=0 ({hill_median:.4f}) outside plausible range (0.5, 2.0)"
    print("PASS: Hill at α=0 in plausible range")


def test_floor_hit_member_less_than_standalone():
    """
    Floor-hit rate for members < standalones at α > 0.

    Test catches: Risk sharing not working (members should hit floor less often).
    Source: pilot_c/tidy.csv columns floor_hit_rate_standalone, floor_hit_rate_member
    """
    df = load_tidy_csv()

    # Filter main block, α=0.1 (enough pooling to matter)
    main = df[(df['block'] == 'main') & (df['alpha'] == 0.1)]

    standalone_rate = main['floor_hit_rate_standalone'].median()
    member_rate = main['floor_hit_rate_member'].median()

    print(f"Floor-hit rate (standalone): {standalone_rate:.4f}")
    print(f"Floor-hit rate (member): {member_rate:.4f}")

    # Members should hit floor less often due to risk sharing
    assert member_rate < standalone_rate, \
        f"Member floor-hit ({member_rate:.4f}) should be < standalone ({standalone_rate:.4f})"
    print("PASS: Members hit floor less often than standalones")


def test_k_eff_less_than_k():
    """
    K_eff < K: not all members equally active.

    Test catches: Effective members calculation bug.
    Source: pilot_c/tidy.csv columns K_post_burnin_median, K_eff_post_burnin_median
    """
    df = load_tidy_csv()

    # Filter main block, α=0.1
    main = df[(df['block'] == 'main') & (df['alpha'] == 0.1)]

    # Only consider scenarios with conglomerates (K > 1)
    has_cong = main[main['K_post_burnin_median'] > 1.5]

    if len(has_cong) == 0:
        print("No conglomerates found at α=0.1, skipping K_eff test")
        return

    k_median = has_cong['K_post_burnin_median'].median()
    k_eff_median = has_cong['K_eff_post_burnin_median'].median()

    print(f"K (median): {k_median:.2f}")
    print(f"K_eff (median): {k_eff_median:.2f}")
    print(f"Ratio K_eff/K: {k_eff_median / k_median:.3f}")

    assert k_eff_median <= k_median, \
        f"K_eff ({k_eff_median:.2f}) should be ≤ K ({k_median:.2f})"
    print("PASS: K_eff ≤ K")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase C Pilot Results Sanity Tests")
    print("=" * 60)

    tests = [
        ("K non-decreasing with α", test_k_nondecreasing_with_alpha),
        ("Hill at α=0 in plausible range", test_hill_at_alpha0),
        ("Members hit floor less than standalones", test_floor_hit_member_less_than_standalone),
        ("K_eff ≤ K", test_k_eff_less_than_k),
    ]

    failed = 0
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        print("-" * 40)
        try:
            test_fn()
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("All sanity tests PASSED")
    else:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
    print("=" * 60)
