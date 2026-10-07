#!/usr/bin/env python3
"""
C10: Burn-in acceptance tests.

Tests read diagnostics/burn_in_c.csv and verify:
1. alpha=0 final-2000 mean hill_exponent in (0.7, 1.5)
2. alpha=0.1 final-2000 mean hill_exponent in (0.5, 1.5) and mean_k > 1 (conglomerates form)
3. No NaN or inf anywhere in the four series at either alpha

Test catches:
- Wrong Hill estimator computation (outside expected range)
- No conglomerate formation at alpha=0.1 (mean_k too low)
- Numerical instability (NaN/inf in metrics)
"""
import numpy as np
import csv
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_burnin_csv(filepath='diagnostics/burn_in_c.csv'):
    """Load burn-in CSV and return data by alpha."""
    data = {'alpha_0.0': {'steps': [], 'hill': [], 'mean_k': [], 'k_eff': [], 'ccs': [], 'fhr': []},
            'alpha_0.1': {'steps': [], 'hill': [], 'mean_k': [], 'k_eff': [], 'ccs': [], 'fhr': []}}

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha_key = f"alpha_{row['alpha']}"
            data[alpha_key]['steps'].append(int(row['step']))
            data[alpha_key]['hill'].append(float(row['hill']))
            data[alpha_key]['mean_k'].append(float(row['mean_k']))
            data[alpha_key]['k_eff'].append(float(row['k_eff']))
            data[alpha_key]['ccs'].append(float(row['ccs']))
            data[alpha_key]['fhr'].append(float(row['fhr']))

    # Convert to numpy arrays
    for alpha_key in data:
        for metric in data[alpha_key]:
            data[alpha_key][metric] = np.array(data[alpha_key][metric])

    return data


def test_alpha0_hill_bounds():
    """
    alpha=0 final-2000 mean hill_exponent in (0.7, 1.5).

    Test catches: Hill estimator bug producing values outside expected range
    for solo firms with reflecting floor.
    """
    data = load_burnin_csv()

    # Get final 2000 steps (metric_every=100, so 20 observations)
    hill = data['alpha_0.0']['hill']
    final_idx = 20  # 2000 / 100
    final_hill = np.nanmean(hill[-final_idx:])

    print(f"alpha=0 final-2000 mean hill_exponent: {final_hill:.4f}")

    assert not np.isnan(final_hill), "alpha=0 hill_exponent is NaN"
    assert 0.7 < final_hill < 1.5, \
        f"alpha=0 hill_exponent = {final_hill:.4f} not in (0.7, 1.5)"
    print("PASS: alpha=0 hill_exponent in bounds")


def test_alpha01_hill_and_mean_k():
    """
    alpha=0.1 final-2000 mean hill_exponent in (0.5, 1.5) and mean_k > 1.

    Test catches:
    - Hill outside range indicating computational error
    - No conglomerate formation (mean_k <= 1 means only solo firms)

    Note: With proportional sharing and reflecting floor, conglomerates form
    but their capital share remains low. We check mean_k > 1 to verify
    M&A activity occurs.
    """
    data = load_burnin_csv()

    # Get final 2000 steps
    final_idx = 20
    hill = data['alpha_0.1']['hill']
    mean_k = data['alpha_0.1']['mean_k']

    final_hill = np.nanmean(hill[-final_idx:])
    final_mean_k = np.nanmean(mean_k[-final_idx:])

    print(f"alpha=0.1 final-2000 mean hill_exponent: {final_hill:.4f}")
    print(f"alpha=0.1 final-2000 mean mean_k: {final_mean_k:.4f}")

    assert not np.isnan(final_hill), "alpha=0.1 hill_exponent is NaN"
    assert 0.5 < final_hill < 1.5, \
        f"alpha=0.1 hill_exponent = {final_hill:.4f} not in (0.5, 1.5)"
    print("PASS: alpha=0.1 hill_exponent in bounds")

    assert not np.isnan(final_mean_k), "alpha=0.1 mean_k is NaN"
    assert final_mean_k > 1.0, \
        f"alpha=0.1 mean_k = {final_mean_k:.4f} <= 1.0 (no conglomerates)"
    print("PASS: alpha=0.1 mean_k > 1.0 (conglomerates form)")


def test_no_nan_inf():
    """
    No NaN or inf anywhere in the four series at either alpha.

    Test catches: Numerical instability in metric computation.
    """
    data = load_burnin_csv()

    metrics = ['hill', 'mean_k', 'k_eff', 'ccs', 'fhr']

    for alpha_key in ['alpha_0.0', 'alpha_0.1']:
        for metric in metrics:
            series = data[alpha_key][metric]
            n_nan = np.sum(np.isnan(series))
            n_inf = np.sum(np.isinf(series))

            assert n_nan == 0, \
                f"{alpha_key} {metric} has {n_nan} NaN values"
            assert n_inf == 0, \
                f"{alpha_key} {metric} has {n_inf} inf values"

            print(f"OK: {alpha_key} {metric} - no NaN/inf")

    print("PASS: No NaN or inf in any series")


if __name__ == '__main__':
    print("=" * 60)
    print("C10 Burn-in Acceptance Tests")
    print("=" * 60)

    print("\nTest 1: alpha=0 Hill exponent bounds")
    print("-" * 40)
    try:
        test_alpha0_hill_bounds()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print("\nTest 2: alpha=0.1 Hill exponent and mean_k")
    print("-" * 40)
    try:
        test_alpha01_hill_and_mean_k()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print("\nTest 3: No NaN/inf in any series")
    print("-" * 40)
    try:
        test_no_nan_inf()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All C10 burn-in acceptance tests PASSED")
    print("=" * 60)
