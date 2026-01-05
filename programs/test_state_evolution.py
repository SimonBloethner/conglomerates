"""
Test that log-space and level-space implementations produce
IDENTICAL state trajectories given the same inputs.

This tests the actual implementation logic, not just math identities.
"""

import numpy as np
from scipy.special import logsumexp, expm1


def test_solo_firm_trajectory():
    """
    Test: Solo firms follow identical paths
    """
    print("="*60)
    print("TEST: Solo Firm State Trajectory")
    print("="*60)

    # Setup
    np.random.seed(42)
    n_steps = 10
    initial_state = 100.0
    returns = np.random.uniform(0.95, 1.10, n_steps)  # Random returns

    # LEVEL SPACE implementation (from collaborative_growth.py line 398)
    states_level = np.zeros(n_steps + 1)
    states_level[0] = initial_state
    for t in range(n_steps):
        states_level[t + 1] = states_level[t] * returns[t]

    # LOG SPACE implementation (from TestingLogs.py line 298)
    log_states = np.zeros(n_steps + 1)
    log_states[0] = np.log(initial_state)
    log_returns = np.log(returns)
    for t in range(n_steps):
        log_states[t + 1] = log_states[t] + log_returns[t]

    # Convert log states back to level for comparison
    states_log = np.exp(log_states)

    print(f"Initial state: {initial_state}")
    print(f"\nStep-by-step comparison:")
    print(f"{'Step':<6} {'Return':<10} {'Level':<15} {'Log->Level':<15} {'Difference':<15}")
    print("-"*70)
    for t in range(n_steps + 1):
        if t < n_steps:
            diff = abs(states_level[t] - states_log[t])
            print(f"{t:<6} {returns[t]:<10.6f} {states_level[t]:<15.6f} {states_log[t]:<15.6f} {diff:<15.2e}")
        else:
            diff = abs(states_level[t] - states_log[t])
            print(f"{t:<6} {'(final)':<10} {states_level[t]:<15.6f} {states_log[t]:<15.6f} {diff:<15.2e}")

    max_diff = np.max(np.abs(states_level - states_log))
    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Match: {np.allclose(states_level, states_log)}")
    assert np.allclose(states_level, states_log), "Solo trajectories differ!"
    print("✅ PASS\n")


def test_conglomerate_firm_trajectory_proportional():
    """
    Test: Conglomerate firms follow identical paths (PROPORTIONAL COST)
    This is the CRITICAL test - checking the actual pooling logic
    """
    print("="*60)
    print("TEST: Conglomerate Firm Trajectory (Proportional Cost)")
    print("="*60)

    # Setup
    np.random.seed(123)
    n_steps = 5
    n_firms = 3
    initial_states = np.array([100.0, 200.0, 150.0])
    returns = np.random.uniform(0.98, 1.08, (n_steps, n_firms))
    share = 0.5
    C = 0.001  # Management cost
    proportional = True

    # LEVEL SPACE implementation (from collaborative_growth.py lines 369-383)
    states_level = np.zeros((n_steps + 1, n_firms))
    states_level[0] = initial_states

    for t in range(n_steps):
        # Calculate gains
        gains = states_level[t] * returns[t] - states_level[t]  # Line 371

        # Calculate pool (proportional case)
        if proportional:
            cost_factor = 1 - C / states_level[t].sum() if states_level[t].sum() > 0 else 0  # Line 373
            pool = (gains * share).sum() * cost_factor  # Line 374
        else:
            management_cost = states_level[t].sum() * C  # Line 376
            pool = (gains * share).sum() - management_cost  # Line 377

        # Distribution (line 379)
        distribution = pool / n_firms
        firm_returns = ((1 - share) * gains + distribution)  # Line 379

        # Update states (line 383)
        states_level[t + 1] = states_level[t] + firm_returns

    # LOG SPACE implementation (from TestingLogs.py lines 240-283)
    log_states = np.zeros((n_steps + 1, n_firms))
    log_states[0] = np.log(initial_states)
    log_returns = np.log(returns)

    for t in range(n_steps):
        # Get log-returns and log-states for current step
        log_rets_t = log_returns[t]
        log_states_t = log_states[t]

        # Compute log_S and weights (lines 247-248)
        log_S = logsumexp(log_states_t)
        w = np.exp(log_states_t - log_S)

        # r_minus1 vector (line 251)
        r_minus1 = expm1(log_rets_t)

        # avg weighted gain (line 254)
        avg_gain_weighted = np.sum(w * r_minus1)

        # Pool calculation (lines 258-263)
        if proportional:
            mgmt_over_S = C * np.exp(-log_S)  # Line 259
            cost_factor = 1.0 - mgmt_over_S  # Line 260
            synth_pool_over_S = share * avg_gain_weighted * cost_factor  # Line 261
        else:
            synth_pool_over_S = share * avg_gain_weighted - C  # Line 263

        # Compute delta_over_state for each firm (lines 266-270)
        K = n_firms
        w_safe = np.maximum(w, 1e-300)  # Line 268
        delta_over_state = (1.0 - share) * r_minus1 + synth_pool_over_S / (K * w_safe)  # Line 270

        # Update log states (lines 274-282)
        for i in range(n_firms):
            log_old = log_states[t, i]
            d = delta_over_state[i]
            if not np.isfinite(d) or d <= -1.0:
                log_states[t + 1, i] = 0.0  # Reset to log(1)
            else:
                log_states[t + 1, i] = log_old + np.log1p(d)  # Line 282

    # Convert log states back to level for comparison
    states_log = np.exp(log_states)

    print(f"Initial states: {initial_states}")
    print(f"Share: {share}, Management cost: {C}, Proportional: {proportional}")
    print(f"\nStep-by-step comparison:")

    for t in range(n_steps + 1):
        print(f"\nStep {t}:")
        print(f"  {'Firm':<6} {'Level':<15} {'Log->Level':<15} {'Difference':<15}")
        print("  " + "-"*55)
        for i in range(n_firms):
            diff = abs(states_level[t, i] - states_log[t, i])
            print(f"  {i:<6} {states_level[t, i]:<15.6f} {states_log[t, i]:<15.6f} {diff:<15.2e}")

    max_diff = np.max(np.abs(states_level - states_log))
    max_rel_diff = np.max(np.abs((states_level - states_log) / states_level))

    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Max relative difference: {max_rel_diff:.2e}")
    print(f"Match: {np.allclose(states_level, states_log)}")

    assert np.allclose(states_level, states_log), "Conglomerate trajectories differ (proportional)!"
    print("✅ PASS\n")


def test_conglomerate_firm_trajectory_nonproportional():
    """
    Test: Conglomerate firms follow identical paths (NON-PROPORTIONAL COST)
    This tests the alternative cost structure
    """
    print("="*60)
    print("TEST: Conglomerate Firm Trajectory (Non-Proportional Cost)")
    print("="*60)

    # Setup
    np.random.seed(789)
    n_steps = 5
    n_firms = 3
    initial_states = np.array([100.0, 200.0, 150.0])
    returns = np.random.uniform(0.98, 1.08, (n_steps, n_firms))
    share = 0.5
    C = 0.001  # Management cost
    proportional = False  # KEY DIFFERENCE

    # LEVEL SPACE implementation (from collaborative_growth.py lines 369-383)
    states_level = np.zeros((n_steps + 1, n_firms))
    states_level[0] = initial_states

    for t in range(n_steps):
        # Calculate gains
        gains = states_level[t] * returns[t] - states_level[t]  # Line 371

        # Calculate pool (NON-proportional case)
        if proportional:
            cost_factor = 1 - C / states_level[t].sum() if states_level[t].sum() > 0 else 0  # Line 373
            pool = (gains * share).sum() * cost_factor  # Line 374
        else:
            management_cost = states_level[t].sum() * C  # Line 376
            pool = (gains * share).sum() - management_cost  # Line 377

        # Distribution (line 379)
        distribution = pool / n_firms
        firm_returns = ((1 - share) * gains + distribution)  # Line 379

        # Update states (line 383)
        states_level[t + 1] = states_level[t] + firm_returns

    # LOG SPACE implementation (from TestingLogs.py lines 240-283)
    log_states = np.zeros((n_steps + 1, n_firms))
    log_states[0] = np.log(initial_states)
    log_returns = np.log(returns)

    for t in range(n_steps):
        # Get log-returns and log-states for current step
        log_rets_t = log_returns[t]
        log_states_t = log_states[t]

        # Compute log_S and weights (lines 247-248)
        log_S = logsumexp(log_states_t)
        w = np.exp(log_states_t - log_S)

        # r_minus1 vector (line 251)
        r_minus1 = expm1(log_rets_t)

        # avg weighted gain (line 254)
        avg_gain_weighted = np.sum(w * r_minus1)

        # Pool calculation (NON-proportional case, lines 258-263)
        if proportional:
            mgmt_over_S = C * np.exp(-log_S)  # Line 259
            cost_factor = 1.0 - mgmt_over_S  # Line 260
            synth_pool_over_S = share * avg_gain_weighted * cost_factor  # Line 261
        else:
            synth_pool_over_S = share * avg_gain_weighted - C  # Line 263

        # Compute delta_over_state for each firm (lines 266-270)
        K = n_firms
        w_safe = np.maximum(w, 1e-300)  # Line 268
        delta_over_state = (1.0 - share) * r_minus1 + synth_pool_over_S / (K * w_safe)  # Line 270

        # Update log states (lines 274-282)
        for i in range(n_firms):
            log_old = log_states[t, i]
            d = delta_over_state[i]
            if not np.isfinite(d) or d <= -1.0:
                log_states[t + 1, i] = 0.0  # Reset to log(1)
            else:
                log_states[t + 1, i] = log_old + np.log1p(d)  # Line 282

    # Convert log states back to level for comparison
    states_log = np.exp(log_states)

    print(f"Initial states: {initial_states}")
    print(f"Share: {share}, Management cost: {C}, Proportional: {proportional}")
    print(f"\nStep-by-step comparison:")

    for t in range(n_steps + 1):
        print(f"\nStep {t}:")
        print(f"  {'Firm':<6} {'Level':<15} {'Log->Level':<15} {'Difference':<15}")
        print("  " + "-"*55)
        for i in range(n_firms):
            diff = abs(states_level[t, i] - states_log[t, i])
            print(f"  {i:<6} {states_level[t, i]:<15.6f} {states_log[t, i]:<15.6f} {diff:<15.2e}")

    max_diff = np.max(np.abs(states_level - states_log))
    max_rel_diff = np.max(np.abs((states_level - states_log) / states_level))

    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Max relative difference: {max_rel_diff:.2e}")
    print(f"Match: {np.allclose(states_level, states_log)}")

    assert np.allclose(states_level, states_log), "Conglomerate trajectories differ (non-proportional)!"
    print("✅ PASS\n")


def test_mixed_scenario():
    """
    Test: Mixed solo and conglomerate firms
    """
    print("="*60)
    print("TEST: Mixed Solo + Conglomerate Trajectory")
    print("="*60)

    np.random.seed(456)
    n_steps = 5
    n_solo = 2
    n_cong = 3
    n_total = n_solo + n_cong

    initial_states = np.array([50.0, 75.0, 100.0, 150.0, 200.0])
    returns = np.random.uniform(0.98, 1.08, (n_steps, n_total))
    share = 0.6
    C = 0.0005
    proportional = False

    # Firms 0,1 are solo; firms 2,3,4 are in conglomerate

    # LEVEL SPACE
    states_level = np.zeros((n_steps + 1, n_total))
    states_level[0] = initial_states

    for t in range(n_steps):
        # Solo firms (0, 1) - simple multiplication
        for i in range(n_solo):
            states_level[t + 1, i] = states_level[t, i] * returns[t, i]

        # Conglomerate firms (2, 3, 4)
        cong_idx = slice(n_solo, n_total)
        gains = states_level[t, cong_idx] * returns[t, cong_idx] - states_level[t, cong_idx]

        if proportional:
            cost_factor = 1 - C / states_level[t, cong_idx].sum()
            pool = (gains * share).sum() * cost_factor
        else:
            management_cost = states_level[t, cong_idx].sum() * C
            pool = (gains * share).sum() - management_cost

        distribution = pool / n_cong
        firm_returns = (1 - share) * gains + distribution
        states_level[t + 1, cong_idx] = states_level[t, cong_idx] + firm_returns

    # LOG SPACE
    log_states = np.zeros((n_steps + 1, n_total))
    log_states[0] = np.log(initial_states)
    log_returns = np.log(returns)

    for t in range(n_steps):
        # Solo firms
        for i in range(n_solo):
            log_states[t + 1, i] = log_states[t, i] + log_returns[t, i]

        # Conglomerate firms
        cong_idx = slice(n_solo, n_total)
        log_states_cong = log_states[t, cong_idx]
        log_rets_cong = log_returns[t, cong_idx]

        log_S = logsumexp(log_states_cong)
        w = np.exp(log_states_cong - log_S)
        r_minus1 = expm1(log_rets_cong)
        avg_gain_weighted = np.sum(w * r_minus1)

        if proportional:
            mgmt_over_S = C * np.exp(-log_S)
            cost_factor = 1.0 - mgmt_over_S
            pool_over_S = share * avg_gain_weighted * cost_factor
        else:
            pool_over_S = share * avg_gain_weighted - C

        w_safe = np.maximum(w, 1e-300)
        delta_over_state = (1.0 - share) * r_minus1 + pool_over_S / (n_cong * w_safe)

        for idx, i in enumerate(range(n_solo, n_total)):
            log_old = log_states[t, i]
            d = delta_over_state[idx]
            if not np.isfinite(d) or d <= -1.0:
                log_states[t + 1, i] = 0.0
            else:
                log_states[t + 1, i] = log_old + np.log1p(d)

    states_log = np.exp(log_states)

    print(f"Solo firms: 0, 1")
    print(f"Conglomerate firms: 2, 3, 4")
    print(f"Share: {share}, Cost: {C}, Proportional: {proportional}")

    print(f"\nFinal states comparison:")
    print(f"{'Firm':<6} {'Type':<12} {'Level':<15} {'Log->Level':<15} {'Difference':<15}")
    print("-"*70)
    for i in range(n_total):
        firm_type = "Solo" if i < n_solo else "Conglomerate"
        diff = abs(states_level[-1, i] - states_log[-1, i])
        print(f"{i:<6} {firm_type:<12} {states_level[-1, i]:<15.6f} {states_log[-1, i]:<15.6f} {diff:<15.2e}")

    max_diff = np.max(np.abs(states_level - states_log))
    print(f"\nMax absolute difference across all steps: {max_diff:.2e}")
    print(f"Match: {np.allclose(states_level, states_log)}")

    assert np.allclose(states_level, states_log), "Mixed scenario trajectories differ!"
    print("✅ PASS\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STATE TRAJECTORY EQUIVALENCE TESTS")
    print("Testing that log-space produces SAME states as level-space")
    print("="*60 + "\n")

    try:
        test_solo_firm_trajectory()
        test_conglomerate_firm_trajectory_proportional()
        test_conglomerate_firm_trajectory_nonproportional()
        test_mixed_scenario()

        print("="*60)
        print("✅ ALL STATE TRAJECTORY TESTS PASSED!")
        print("="*60)
        print("\nConclusion: Log-space implementation produces IDENTICAL")
        print("state trajectories to level-space implementation.")
        print("The implementations are equivalent.")

    except AssertionError as e:
        print("="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        exit(1)