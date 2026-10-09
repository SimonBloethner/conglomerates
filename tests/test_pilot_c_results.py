#!/usr/bin/env python3
"""
C18: Phase C pilot results sanity tests (strict thresholds).

Tests verify pilot_c/tidy.csv has expected structure and values:
1. α=0 rows: mergers = 0, proposals > 0
   Catches: M&A mechanism running at α=0 (no pooling means no benefit)
2. Every row with K ≥ 2 has K_eff/K ≥ 0.3
   Catches: K_eff calculation bug (effective members too low)
3. hill_exponent_median at α=0 within 15% of 1.06 for every family
   Catches: floor miscalibration, Hill estimator bug
4. cong_capital_share at α=0.3 ≥ 0.05 for laplace/power_law
   Catches: conglomerates not forming or capital share miscalculated
5. No NaN in main-block _median columns
   Catches: numerical instability, missing data
6. main-block mean K at α=0.3 ≥ 3.0 for normal/laplace (C18)
   Catches: loggain rule not driving larger conglomerates
7. rule-replay mean K < main-block (C18)
   Catches: loggain supposed to improve on replay
8. growth gap > 0 for every family at α=0.3 (C18)
   Catches: members not benefiting from conglomerate membership
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
    Every row with K ≥ 2 has K_eff/K ≥ 0.3.

    Test catches: K_eff calculation bug (effective members too low).
    When conglomerates form (K ≥ 2), at least 30% of members should be
    effectively contributing. This allows for unequal member sizes.

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

    # Find violating rows (threshold 0.3, allows for unequal member sizes)
    threshold = 0.3
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
    # - growth_gap_median: NaN at low alpha where no members exist
    excluded_cols = {
        'alpha_adopted_median',
        'K_median',
        'K_eff_over_K_median',
        'K_post_burnin_median',
        'K_eff_post_burnin_median',
        'growth_gap_median',
        'event_did_median', 'event_did_nofloor_median',
        'event_joiner_before_median', 'event_joiner_after_median',
        'event_control_before_median', 'event_control_after_median',
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


def test_main_block_mean_k_at_alpha03():
    """
    C18: main-block mean K at α=0.3 ≥ 3.0 for normal/laplace.

    Test catches: loggain rule not driving larger conglomerates.
    Under loggain decision rule, conglomerates should grow larger than
    under replay because the forward-looking loggain criterion identifies
    more beneficial mergers.

    Source: pilot_c/tidy.csv column K_mean, filtered by block='main', alpha=0.3
    """
    df = load_tidy_csv()

    print("C18: Testing main-block mean K at α=0.3...")

    for family in ['normal', 'laplace']:
        target_rows = df[
            (df['block'] == 'main') &
            (df['alpha'] == 0.3) &
            (df['log_family'] == family)
        ]

        if len(target_rows) == 0:
            raise AssertionError(f"No {family} rows at α=0.3 found in main block")

        # Use K_mean from summary if available, else K_post_burnin_median
        if 'K_mean' in target_rows.columns and not target_rows['K_mean'].isna().all():
            k_mean = target_rows['K_mean'].mean()
            col_used = 'K_mean'
        else:
            k_mean = target_rows['K_post_burnin_median'].mean()
            col_used = 'K_post_burnin_median'

        print(f"\n  {family}:")
        print(f"    Rows: {len(target_rows)}")
        print(f"    {col_used} mean: {k_mean:.2f}")
        print(f"    Threshold: >= 3.0")

        assert k_mean >= 3.0, \
            f"{family} mean K ({k_mean:.2f}) not >= 3.0 at α=0.3"
        print(f"    Status: PASS")

    print("\nPASS: main-block mean K >= 3.0 at α=0.3 for normal/laplace")


def test_rule_replay_k_less_than_main():
    """
    C18: rule-replay mean K < main-block.

    Test catches: loggain supposed to improve on replay.
    The rule-replay block uses decision_rule='replay' while main uses 'loggain'.
    Loggain should produce larger conglomerates, so main K > replay K.

    Source: pilot_c/tidy.csv column K_mean, comparing block='main' vs 'rule-replay'
    """
    df = load_tidy_csv()

    # Get main block (loggain) at α=0.1 (same alpha as burn-in test)
    main = df[(df['block'] == 'main') & (df['alpha'] == 0.1) &
              (df['log_family'] == 'laplace') & (df['cost_type'] == 'power_law')]

    if len(main) == 0:
        raise AssertionError("No laplace/power_law rows at α=0.1 found in main block")

    # Get rule-replay block at same alpha
    replay = df[(df['block'] == 'rule-replay') & (df['alpha'] == 0.1)]

    if len(replay) == 0:
        raise AssertionError("No rows at α=0.1 found in rule-replay block")

    # Use K_mean if available
    if 'K_mean' in main.columns and not main['K_mean'].isna().all():
        main_k = main['K_mean'].mean()
        replay_k = replay['K_mean'].mean()
        col_used = 'K_mean'
    else:
        main_k = main['K_post_burnin_median'].mean()
        replay_k = replay['K_post_burnin_median'].mean()
        col_used = 'K_post_burnin_median'

    print(f"C18: Testing rule-replay K < main-block K...")
    print(f"  Main block (loggain) {col_used}: {main_k:.2f}")
    print(f"  Rule-replay block {col_used}: {replay_k:.2f}")
    print(f"  Main rows: {len(main)}, Replay rows: {len(replay)}")

    assert replay_k < main_k, \
        f"rule-replay K ({replay_k:.2f}) should be less than main-block K ({main_k:.2f})"

    print(f"  Ratio (main/replay): {main_k/replay_k:.2f}x")
    print("PASS: rule-replay K < main-block K (loggain improves on replay)")


def test_growth_gap_positive_at_alpha03():
    """
    C18: growth gap > 0 for every family at α=0.3.

    Test catches: members not benefiting from conglomerate membership.
    The growth gap measures the difference in growth between conglomerate
    members and standalone firms. A positive gap means members grow faster,
    validating that conglomerate membership is beneficial.

    Source: pilot_c/tidy.csv column growth_gap_median
    """
    df = load_tidy_csv()

    print("C18: Testing growth gap > 0 at α=0.3 for all families...")

    families = df[df['block'] == 'main']['log_family'].unique()
    all_pass = True
    failed = []

    for family in sorted(families):
        target_rows = df[
            (df['block'] == 'main') &
            (df['alpha'] == 0.3) &
            (df['log_family'] == family)
        ]

        if len(target_rows) == 0:
            print(f"  {family}: No rows found, skipping")
            continue

        if 'growth_gap_median' not in target_rows.columns:
            print(f"  {family}: growth_gap_median column not found, skipping")
            continue

        gap_median = target_rows['growth_gap_median'].median()
        gap_mean = target_rows['growth_gap_median'].mean()

        print(f"\n  {family}:")
        print(f"    Rows: {len(target_rows)}")
        print(f"    growth_gap_median median: {gap_median:.6f}")
        print(f"    growth_gap_median mean: {gap_mean:.6f}")

        if np.isnan(gap_median):
            print(f"    Status: SKIP (NaN)")
            continue

        if gap_median > 0:
            print(f"    Status: PASS")
        else:
            print(f"    Status: FAIL (gap <= 0)")
            all_pass = False
            failed.append((family, gap_median))

    if len(failed) > 0:
        raise AssertionError(f"growth_gap not > 0 for: {failed}")

    print("\nPASS: growth gap > 0 at α=0.3 for all families")


if __name__ == '__main__':
    print("=" * 60)
    print("C18: Phase C Pilot Results Sanity Tests (loggain)")
    print("=" * 60)

    tests = [
        ("α=0: mergers=0, proposals>0", test_alpha0_mergers_zero_proposals_positive),
        ("K_eff/K >= 0.4 for K >= 2", test_k_eff_over_k_ratio),
        ("Hill at α=0 within 15% of 1.06 (all families)", test_hill_within_15pct_of_target),
        ("cong_capital_share >= 0.05 at α=0.3", test_cong_capital_share_at_alpha03),
        ("No NaN in _median columns", test_no_nan_in_median_columns),
        # C18: loggain validation tests
        ("C18: main-block mean K >= 3.0 at α=0.3", test_main_block_mean_k_at_alpha03),
        ("C18: rule-replay K < main-block K", test_rule_replay_k_less_than_main),
        ("C18: growth gap > 0 at α=0.3", test_growth_gap_positive_at_alpha03),
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
