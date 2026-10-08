#!/usr/bin/env python3
"""
C19: Performance tests for the hoisted loggain implementation.

Tests:
1. Bit-identical: pre-hoist reference vs post-hoist produce identical results
2. Speedup: 1000 calls at K=15, h=500 should be at least 5x faster
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collaborative_growth as cg
from tests._member_gain_reference import member_gain_loggain_reference


def test_bit_identical():
    """
    Verify that hoisted functions produce identical results to reference.

    Test catches: hoisting that introduces numerical differences.
    """
    np.random.seed(42)
    cg.seed_numba(42)

    # M=N=20, T=600, alpha=0.3 as specified
    M = N = 20
    T = 600
    alpha = 0.3
    g = 0.02
    iqr = 0.2
    scale = iqr / 1.3489795003921634

    # Test multiple scenarios
    scenarios = [
        (2, 50),   # K=2, h=50
        (5, 100),  # K=5, h=100
        (10, 200), # K=10, h=200
        (15, 500), # K=15, h=500 (main perf case)
    ]

    all_match = True
    for K, h in scenarios:
        # Generate test data
        np.random.seed(42 + K * 100)
        returns = np.random.normal(g, scale, (h, K))
        log_states = np.zeros((h + 1, K))
        for t in range(h):
            log_states[t + 1, :] = log_states[t, :] + returns[t, :]
        sizes = np.exp(log_states[:-1, :])

        members = np.arange(K)
        Phi_K = cg.management_cost_function(K, 'power_law')

        for sharing_rule in [0, 1]:  # equal, proportional
            for i in range(K):
                # Reference (pre-hoist)
                gain_ref = member_gain_loggain_reference(
                    i, members, alpha, returns, sizes, Phi_K, sharing_rule, g
                )

                # New hoisted version using the convenience wrapper
                gain_new = cg.member_gain_loggain(
                    i, members, alpha, returns, sizes, Phi_K, sharing_rule, g
                )

                if not np.isclose(gain_ref, gain_new, rtol=0, atol=0):
                    print(f"MISMATCH at K={K}, h={h}, i={i}, rule={sharing_rule}:")
                    print(f"  reference: {gain_ref}")
                    print(f"  new:       {gain_new}")
                    print(f"  diff:      {abs(gain_ref - gain_new)}")
                    all_match = False

    assert all_match, "Hoisted implementation produces different results"
    print("PASS: Bit-identical results (pre-hoist == post-hoist)")


def test_speedup():
    """
    Verify that hoisted version is at least 5x faster at K=15, h=500.

    Test catches: a hoist that doesn't actually hoist.
    """
    np.random.seed(42)
    cg.seed_numba(42)

    K = 15
    h = 500
    alpha = 0.3
    g = 0.02
    iqr = 0.2
    scale = iqr / 1.3489795003921634

    # Generate test data
    returns = np.random.normal(g, scale, (h, K))
    log_states = np.zeros((h + 1, K))
    for t in range(h):
        log_states[t + 1, :] = log_states[t, :] + returns[t, :]
    sizes = np.exp(log_states[:-1, :])

    members = np.arange(K)
    Phi_K = cg.management_cost_function(K, 'power_law')
    sharing_rule = 1  # proportional

    n_calls = 1000

    # Warm up JIT
    for _ in range(10):
        member_gain_loggain_reference(0, members, alpha, returns, sizes, Phi_K, sharing_rule, g)
        cg.prepare_window_loggain(members, returns, sizes, g)
        cg.member_gain_from_prepared(0, K, *cg.prepare_window_loggain(members, returns, sizes, g),
                                     alpha, Phi_K, sharing_rule)

    # Time reference (one-shot per member, must prepare for each)
    start = time.perf_counter()
    for _ in range(n_calls):
        for i in range(K):
            member_gain_loggain_reference(i, members, alpha, returns, sizes, Phi_K, sharing_rule, g)
    time_ref = time.perf_counter() - start

    # Time hoisted (prepare once, evaluate K times)
    start = time.perf_counter()
    for _ in range(n_calls):
        eps, sz, sum_s, sum_s_eps = cg.prepare_window_loggain(members, returns, sizes, g)
        for i in range(K):
            cg.member_gain_from_prepared(i, K, eps, sz, sum_s, sum_s_eps, alpha, Phi_K, sharing_rule)
    time_hoisted = time.perf_counter() - start

    speedup = time_ref / time_hoisted

    print(f"\nSpeedup test (K={K}, h={h}, {n_calls} calls):")
    print(f"  Reference time:  {time_ref:.3f}s")
    print(f"  Hoisted time:    {time_hoisted:.3f}s")
    print(f"  Speedup:         {speedup:.1f}x")

    assert speedup >= 5.0, f"Speedup {speedup:.1f}x < 5x required"
    print(f"PASS: Speedup {speedup:.1f}x >= 5x")


def test_full_model_identical():
    """
    Run a short simulation with loggain and verify results are deterministic.

    This ensures the hoisted functions integrate correctly into the model.
    """
    np.random.seed(42)
    cg.seed_numba(42)

    M = N = 20
    T = 600
    alpha = 0.3

    params = [
        M, N, T, alpha, M * N,
        0.1, 4, 0.0, False, 50,
        'power_law', None, None, None,
    ]

    # Run twice with same seed
    result1 = cg.model(
        params, seed=42,
        growth_process='log_family', log_family='laplace',
        sigma_range=(0.1, 0.3), floor_c=0.12717,
        g=0.02, market_size_fixed=True,
        decision_rule='loggain'
    )

    result2 = cg.model(
        params, seed=42,
        growth_process='log_family', log_family='laplace',
        sigma_range=(0.1, 0.3), floor_c=0.12717,
        g=0.02, market_size_fixed=True,
        decision_rule='loggain'
    )

    # Compare key outputs
    hyper1 = result1[-1]
    hyper2 = result2[-1]

    # Check Hill exponent arrays
    hill1 = hyper1['hill_exponent']
    hill2 = hyper2['hill_exponent']
    assert np.allclose(hill1, hill2), "Hill exponent differs between runs"

    # Check summary
    s1 = hyper1['summary']
    s2 = hyper2['summary']
    for key in s1:
        if isinstance(s1[key], (int, float)) and not np.isnan(s1[key]):
            assert s1[key] == s2[key], f"Summary {key} differs: {s1[key]} vs {s2[key]}"

    print("PASS: Full model produces deterministic results with hoisted functions")


if __name__ == '__main__':
    print("=" * 60)
    print("C19: Loggain Performance Tests")
    print("=" * 60)

    tests = [
        ("Bit-identical", test_bit_identical),
        ("Speedup (5x)", test_speedup),
        ("Full model deterministic", test_full_model_identical),
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
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("All loggain performance tests PASSED")
    else:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
    print("=" * 60)
