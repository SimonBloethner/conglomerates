#!/usr/bin/env python3
"""
Tests for C14: Burn-in rerun with market_size_fixed and c_star.

Tests verify:
1. α=0 final-2000 mean hill_exponent within 15% of 1.06
   Catches: broken floor reflection, wrong c_star calibration
2. α=0.1: cong_capital_share > 0.02, mean K ≥ 2.5, K_eff/K ≥ 0.5
   Catches: conglomerates not forming or being ineffective
3. No NaN/inf anywhere
   Catches: numerical instability, division by zero
"""
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_burnin_csv(filepath=None):
    """Load burn-in CSV."""
    if filepath is None:
        # Default to diagnostics/burn_in_c.csv relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(project_root, 'diagnostics', 'burn_in_c.csv')
    return pd.read_csv(filepath)


def test_hill_within_15pct_of_target():
    """
    α=0 final-2000 mean hill_exponent within 15% of 1.06.

    Catches: broken floor reflection, wrong c_star calibration.

    The floor at c_star should produce a Pareto tail with exponent ~1.06
    (Axtell 2001 empirical value). With market_size_fixed=True and
    proper floor_c calibration, this should hold.
    """
    df = load_burnin_csv()

    # Filter to alpha=0
    alpha0 = df[df['alpha'] == 0.0]

    if len(alpha0) == 0:
        raise AssertionError("No alpha=0 results found in burn_in_c.csv")

    # Final 2000 steps: with metric_every=100, that's the last 20 observations
    # Total steps = 8000, so steps 6100 to 8000
    final_2000 = alpha0[alpha0['step'] > 6000]

    if len(final_2000) == 0:
        raise AssertionError("No steps > 6000 found for alpha=0")

    hill_mean = final_2000['hill'].mean()
    target = 1.06
    tolerance = 0.15

    print(f"α=0 Hill exponent (final 2000 steps):")
    print(f"  Mean: {hill_mean:.4f}")
    print(f"  Target: {target:.4f}")
    print(f"  Gap: {100 * abs(hill_mean - target) / target:.2f}%")
    print(f"  Tolerance: {100 * tolerance:.0f}%")

    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)

    assert lower <= hill_mean <= upper, \
        f"Hill ({hill_mean:.4f}) not within 15% of {target}: expected [{lower:.4f}, {upper:.4f}]"

    print("PASS: α=0 Hill within 15% of 1.06")


def test_alpha01_conglomerate_activity():
    """
    α=0.1: cong_capital_share > 0.02, mean K ≥ 2.5, K_eff/K ≥ 0.5.

    Catches: conglomerates not forming or being ineffective.

    With α=0.1 (10% pooling), conglomerates should:
    - Have meaningful capital share (> 2%)
    - Have at least 2.5 members on average
    - Be at least 50% effective (K_eff/K ≥ 0.5)
    """
    df = load_burnin_csv()

    # Filter to alpha=0.1
    alpha01 = df[df['alpha'] == 0.1]

    if len(alpha01) == 0:
        raise AssertionError("No alpha=0.1 results found in burn_in_c.csv")

    # Final 2000 steps
    final_2000 = alpha01[alpha01['step'] > 6000]

    if len(final_2000) == 0:
        raise AssertionError("No steps > 6000 found for alpha=0.1")

    ccs_mean = final_2000['ccs'].mean()
    mean_k_mean = final_2000['mean_k'].mean()
    k_eff_mean = final_2000['k_eff'].mean()
    k_ratio = k_eff_mean / mean_k_mean if mean_k_mean > 0 else 0.0

    print(f"α=0.1 Conglomerate metrics (final 2000 steps):")
    print(f"  cong_capital_share: {ccs_mean:.4f}")
    print(f"  mean K: {mean_k_mean:.2f}")
    print(f"  K_eff: {k_eff_mean:.2f}")
    print(f"  K_eff/K: {k_ratio:.4f}")

    # Check assertions
    assert ccs_mean > 0.02, \
        f"cong_capital_share ({ccs_mean:.4f}) not > 0.02"
    print("  cong_capital_share > 0.02: PASS")

    assert mean_k_mean >= 2.5, \
        f"mean K ({mean_k_mean:.2f}) not >= 2.5"
    print("  mean K >= 2.5: PASS")

    assert k_ratio >= 0.5, \
        f"K_eff/K ({k_ratio:.4f}) not >= 0.5"
    print("  K_eff/K >= 0.5: PASS")

    print("PASS: α=0.1 conglomerate activity assertions")


def test_no_nan_inf():
    """
    No NaN/inf anywhere.

    Catches: numerical instability, division by zero.
    """
    df = load_burnin_csv()

    # Check all numeric columns
    numeric_cols = ['hill', 'mean_k', 'k_eff', 'ccs', 'fhr_standalone', 'fhr_member', 'growth_gap']

    for col in numeric_cols:
        if col in df.columns:
            has_nan = df[col].isna().any()
            has_inf = np.isinf(df[col]).any()

            if has_nan:
                nan_count = df[col].isna().sum()
                raise AssertionError(f"Column '{col}' has {nan_count} NaN values")

            if has_inf:
                inf_count = np.isinf(df[col]).sum()
                raise AssertionError(f"Column '{col}' has {inf_count} inf values")

    print("No NaN or inf values found in any column")
    print("PASS: No NaN/inf anywhere")


if __name__ == '__main__':
    print("=" * 60)
    print("C14: Burn-in Tests (test_burnin_c3.py)")
    print("=" * 60)

    tests = [
        ("Hill within 15% of 1.06 (α=0)", test_hill_within_15pct_of_target),
        ("Conglomerate activity (α=0.1)", test_alpha01_conglomerate_activity),
        ("No NaN/inf anywhere", test_no_nan_inf),
    ]

    failed = 0
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        print("-" * 40)
        try:
            test_fn()
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
            print("Run run_burnin_c14.py first to generate results.")
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
