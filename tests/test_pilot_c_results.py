#!/usr/bin/env python3
"""
C15b: Phase C pilot results sanity tests (strict thresholds).

Tests verify pilot_c/tidy.csv has expected structure and values:
1. α=0 rows: mergers = 0, proposals > 0
   Catches: M&A mechanism running at α=0 (no pooling means no benefit)
2. Every row with K ≥ 2 has K_eff/K ≥ 0.4
   Catches: K_eff calculation bug (effective members too low)
3. hill_exponent_median at α=0 within 15% of 1.06 for every family
   Catches: floor miscalibration, Hill estimator bug
4. cong_capital_share at α=0.3 ≥ 0.05 for laplace/power_law
   Catches: conglomerates not forming or capital share miscalculated
5. No NaN in main-block _median columns
   Catches: numerical instability, missing data
"""
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tidy_csv(filepath='pilot_c/tidy.csv'):
    """Load tidy CSV."""
    return pd.read_csv(filepath)


def load_medians_csv(filepath='pilot_c/medians.csv'):
    """Load medians CSV."""
    return pd.read_csv(filepath)


def test_alpha0_mergers_zero_proposals_positive():
    """
    α=0 rows: mergers = 0, proposals > 0.

    Test catches: M&A mechanism running at α=0 (no pooling means no benefit).
    At α=0 there's no pooling so mergers should never complete, but proposals
    should still occur (just all rejected).

    Source: pilot_c/tidy.csv columns mergers_per_period, proposals_per_period
    """
    df = load_tidy_csv()

    # Filter α=0 rows in main block
    alpha0 = df[(df['alpha'] == 0.0) & (df['block'] == 'main')]

    if len(alpha0) == 0:
        raise AssertionError("No α=0 rows found in main block")

    # Check mergers = 0 (or very close to 0)
    mergers = alpha0['mergers_per_period']
    max_mergers = mergers.max()

    print(f"α=0 rows: {len(alpha0)}")
    print(f"  mergers_per_period max: {max_mergers:.6f}")
    print(f"  mergers_per_period mean: {mergers.mean():.6f}")

    # Allow tiny numerical tolerance
    assert max_mergers < 1e-6, \
        f"mergers_per_period at α=0 should be 0, got max={max_mergers:.6f}"
    print("  mergers = 0: PASS")

    # Check proposals > 0
    proposals = alpha0['proposals_per_period']
    min_proposals = proposals.min()

    print(f"  proposals_per_period min: {min_proposals:.6f}")
    print(f"  proposals_per_period mean: {proposals.mean():.6f}")

    assert min_proposals > 0, \
        f"proposals_per_period at α=0 should be > 0, got min={min_proposals:.6f}"
    print("  proposals > 0: PASS")

    print("PASS: α=0 mergers=0, proposals>0")


def test_k_eff_over_k_ratio():
    """
    Every row with K ≥ 2 has K_eff/K ≥ 0.4.

    Test catches: K_eff calculation bug (effective members too low).
    When conglomerates form (K ≥ 2), at least 40% of members should be
    effectively contributing. This is a strict threshold with no outlier allowance.

    Source: pilot_c/tidy.csv columns K_post_burnin_median, K_eff_post_burnin_median
    """
    df = load_tidy_csv()

    # Filter rows with K >= 2
    has_cong = df[df['K_post_burnin_median'] >= 2.0].copy()

    if len(has_cong) == 0:
        print("No rows with K >= 2 found, skipping K_eff/K test")
        return

    # Compute ratio
    has_cong['k_ratio'] = has_cong['K_eff_post_burnin_median'] / has_cong['K_post_burnin_median']

    min_ratio = has_cong['k_ratio'].min()
    mean_ratio = has_cong['k_ratio'].mean()

    print(f"Rows with K >= 2: {len(has_cong)}")
    print(f"  K_eff/K min: {min_ratio:.4f}")
    print(f"  K_eff/K mean: {mean_ratio:.4f}")

    # Find violating rows (threshold 0.4, strict)
    threshold = 0.4
    violating = has_cong[has_cong['k_ratio'] < threshold]

    if len(violating) > 0:
        print(f"  Violating rows (< {threshold}): {len(violating)}")
        sample = violating.head(3)
        for _, row in sample.iterrows():
            print(f"    scenario {row['scenario_id']}: α={row['alpha']}, K={row['K_post_burnin_median']:.2f}, "
                  f"K_eff={row['K_eff_post_burnin_median']:.2f}, ratio={row['k_ratio']:.4f}")

    # Strict: no violations allowed
    assert len(violating) == 0, \
        f"Found {len(violating)} rows with K_eff/K < {threshold}"

    print(f"PASS: All rows with K >= 2 have K_eff/K >= {threshold}")


def test_hill_within_15pct_of_target():
    """
    hill_exponent_median at α=0 within 15% of 1.06 for every family.

    Test catches: floor miscalibration (c_star), Hill estimator bug.
    The Axtell (2001) empirical value is ~1.06 and with proper floor calibration
    (floor_c=0.12717 from C14) our simulation should achieve this for all families.

    Source: pilot_c/tidy.csv column hill_exponent_median, filtered by alpha=0
    """
    df = load_tidy_csv()

    # Filter main block, α=0
    main_a0 = df[(df['block'] == 'main') & (df['alpha'] == 0.0)]

    if len(main_a0) == 0:
        raise AssertionError("No α=0 rows found in main block")

    target = 1.06
    tolerance = 0.15  # 15% tolerance
    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)

    print(f"Target Hill: {target:.4f}")
    print(f"Tolerance: ±{tolerance * 100:.0f}% -> [{lower:.4f}, {upper:.4f}]")

    # Check each family (all must pass)
    families = main_a0['log_family'].unique()
    all_pass = True
    failed_families = []

    for family in sorted(families):
        family_data = main_a0[main_a0['log_family'] == family]
        hill_median = family_data['hill_exponent_median'].median()
        gap_pct = 100 * abs(hill_median - target) / target

        print(f"\n  {family}:")
        print(f"    Hill median: {hill_median:.4f}")
        print(f"    Gap from 1.06: {gap_pct:.2f}%")

        if lower <= hill_median <= upper:
            print(f"    Status: PASS")
        else:
            print(f"    Status: FAIL (outside [{lower:.4f}, {upper:.4f}])")
            all_pass = False
            failed_families.append((family, hill_median, gap_pct))

    # All families must pass
    assert all_pass, \
        f"hill_exponent_median at α=0 not within 15% of 1.06 for: {failed_families}"

    print("\nPASS: Hill at α=0 within 15% of 1.06 for all families")


def test_cong_capital_share_at_alpha03():
    """
    cong_capital_share at α=0.3 ≥ 0.05 for laplace/power_law.

    Test catches: conglomerates not forming or capital share miscalculated.
    At 30% pooling (α=0.3), conglomerates should have meaningful capital share.

    Source: pilot_c/tidy.csv column cong_capital_share_median
    """
    df = load_tidy_csv()

    # Filter main block, α=0.3, laplace, power_law
    target_rows = df[
        (df['block'] == 'main') &
        (df['alpha'] == 0.3) &
        (df['log_family'] == 'laplace') &
        (df['cost_type'] == 'power_law')
    ]

    if len(target_rows) == 0:
        raise AssertionError("No laplace/power_law rows at α=0.3 found in main block")

    ccs_median = target_rows['cong_capital_share_median'].median()
    ccs_min = target_rows['cong_capital_share_median'].min()

    print(f"α=0.3, laplace/power_law (main block):")
    print(f"  Rows: {len(target_rows)}")
    print(f"  cong_capital_share_median median: {ccs_median:.4f}")
    print(f"  cong_capital_share_median min: {ccs_min:.4f}")
    print(f"  Threshold: >= 0.05")

    # Also report K to confirm conglomerate activity
    k_mean = target_rows['K_post_burnin_median'].mean()
    print(f"  K_post_burnin_median mean: {k_mean:.2f}")

    assert ccs_min >= 0.05, \
        f"cong_capital_share_median ({ccs_min:.4f}) not >= 0.05 at α=0.3"

    print("PASS: cong_capital_share >= 0.05 at α=0.3 for laplace/power_law")


def test_no_nan_in_median_columns():
    """
    No NaN in _median columns of the main block.

    Test catches: numerical instability, missing data.

    Source: pilot_c/tidy.csv all columns ending in _median
    """
    df = load_tidy_csv()

    # Filter main block
    main = df[df['block'] == 'main']

    if len(main) == 0:
        raise AssertionError("No rows found in main block")

    # Find all _median columns, excluding columns that are legitimately NaN in some rows:
    # - alpha_adopted_median: only for endogenous block
    # - K-related columns: NaN at low alpha where no conglomerates form
    excluded_cols = {
        'alpha_adopted_median',
        'K_median',
        'K_eff_over_K_median',
        'K_post_burnin_median',
        'K_eff_post_burnin_median',
    }
    median_cols = [col for col in main.columns
                   if col.endswith('_median') and col not in excluded_cols]

    print(f"Main block rows: {len(main)}")
    print(f"Checking {len(median_cols)} _median columns...")

    nan_found = []

    for col in median_cols:
        nan_count = main[col].isna().sum()
        if nan_count > 0:
            nan_found.append((col, nan_count))
            print(f"  {col}: {nan_count} NaN values")

    if len(nan_found) == 0:
        print("  No NaN values found")

    assert len(nan_found) == 0, \
        f"Found NaN in main block _median columns: {nan_found}"

    print("\nPASS: No NaN in _median columns of main block")


if __name__ == '__main__':
    print("=" * 60)
    print("C15b: Phase C Pilot Results Sanity Tests (Strict)")
    print("=" * 60)

    tests = [
        ("α=0: mergers=0, proposals>0", test_alpha0_mergers_zero_proposals_positive),
        ("K_eff/K >= 0.4 for K >= 2", test_k_eff_over_k_ratio),
        ("Hill at α=0 within 15% of 1.06 (all families)", test_hill_within_15pct_of_target),
        ("cong_capital_share >= 0.05 at α=0.3", test_cong_capital_share_at_alpha03),
        ("No NaN in _median columns", test_no_nan_in_median_columns),
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("All sanity tests PASSED")
    else:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
    print("=" * 60)
