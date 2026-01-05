"""
Unit test: Compare Numba pooling implementation vs Python pooling implementation.

Tests only the pooling logic in isolation with synthetic conglomerate data.
"""
import numpy as np
from scipy.special import logsumexp, expm1
import numba as nb

# ============================================================================
# NUMBA VERSION (from collaborative_growth.py)
# ============================================================================

@nb.njit
def process_conglomerate_pooling_numba(
    cong_firms,
    cong_size,
    active_cong_ids,
    log_returns,
    log_states_curr,
    management_costs_lookup,
    share,
    proportional,
    lookback,
    step
):
    """
    Numba-compiled pooling function for all active conglomerates.
    """
    n_congs = len(active_cong_ids)
    total_firms = log_states_curr.shape[0]

    # Output arrays
    new_log_states = np.copy(log_states_curr)
    cong_pools = np.zeros(n_congs, dtype=np.float64)

    # Track firms to exit (worst case: all firms could exit)
    exit_list = np.zeros(total_firms, dtype=np.int64)
    exit_count = 0

    # Process each active conglomerate
    for idx in range(n_congs):
        cong_id = active_cong_ids[idx]
        n_firms = cong_size[cong_id]

        if n_firms == 0:
            continue

        # Extract firms in this conglomerate
        conglomerate = cong_firms[cong_id, :n_firms]

        # Get log returns and states for this conglomerate
        log_ret = log_returns[conglomerate]
        log_st = log_states_curr[conglomerate]

        # Step 1: Compute log_S using manual logsumexp
        max_log_st = np.max(log_st)
        sum_exp = 0.0
        for i in range(n_firms):
            sum_exp += np.exp(log_st[i] - max_log_st)
        log_S = max_log_st + np.log(sum_exp)

        # Step 2: Compute normalized weights
        w = np.exp(log_st - log_S)

        # Step 3: Compute r_minus1 = exp(log_returns) - 1
        r_m1 = np.expm1(log_ret)

        # Step 4: Weighted average gain (CORRECTED - no extra division)
        avg_gain_weighted = 0.0
        for i in range(n_firms):
            avg_gain_weighted += w[i] * r_m1[i]

        # Step 5: Management cost
        m = management_costs_lookup[n_firms]

        # Step 6: Pool calculation
        if proportional:
            mgmt_over_S = m * np.exp(-log_S)
            cost_factor = 1.0 - mgmt_over_S
            pool_over_S = share * avg_gain_weighted * cost_factor
        else:
            pool_over_S = share * avg_gain_weighted - m

        # Store pool value
        cong_pools[idx] = pool_over_S

        # Step 7: Compute delta_over_state for each firm
        K = n_firms
        for i in range(n_firms):
            firm = conglomerate[i]
            w_safe = max(w[i], 1e-300)
            delta = (1.0 - share) * r_m1[i] + pool_over_S / (K * w_safe)

            # Check for exit condition
            if not np.isfinite(delta) or delta <= -1.0:
                new_log_states[firm] = 0.0  # Reset to log(1)
                exit_list[exit_count] = firm
                exit_count += 1
            else:
                log_old = log_states_curr[firm]
                new_log_states[firm] = log_old + np.log1p(delta)

    # Return only the firms that need to exit
    firms_to_exit = exit_list[:exit_count]

    return new_log_states, cong_pools, firms_to_exit


# ============================================================================
# PYTHON VERSION (original implementation)
# ============================================================================

def process_conglomerate_pooling_python(
    cong_firms,
    cong_size,
    active_cong_ids,
    log_returns,
    log_states_curr,
    management_costs_lookup,
    share,
    proportional,
    lookback,
    step
):
    """
    Python version of pooling function (original implementation).
    """
    total_firms = log_states_curr.shape[0]
    n_congs = len(active_cong_ids)

    # Output arrays
    new_log_states = np.copy(log_states_curr)
    cong_pools = np.zeros(n_congs, dtype=np.float64)
    firms_to_exit = []

    # Process each active conglomerate
    for idx, cong_id in enumerate(active_cong_ids):
        # Extract firm list from SoA
        n_firms = cong_size[cong_id]
        conglomerate = cong_firms[cong_id, :n_firms]

        # Get current states and returns
        log_returns_cong = log_returns[conglomerate]
        log_states_cong = log_states_curr[conglomerate]

        # Compute log_S and weights
        log_S = logsumexp(log_states_cong)
        w = np.exp(log_states_cong - log_S)

        # r_minus1 vector
        r_minus1 = expm1(log_returns_cong)

        # Average weighted gain
        avg_gain_weighted = np.sum(w * r_minus1)

        m = management_costs_lookup[len(conglomerate)]

        # Pool calculation
        if proportional:
            mgmt_over_S = m * np.exp(-log_S)
            cost_factor = 1.0 - mgmt_over_S
            pool_over_S = share * avg_gain_weighted * cost_factor
        else:
            pool_over_S = share * avg_gain_weighted - m

        # Store pool value
        cong_pools[idx] = pool_over_S

        # Compute delta_over_state for each firm
        K = len(conglomerate)
        w_safe = np.maximum(w, 1e-300)
        delta_over_state = (1.0 - share) * r_minus1 + pool_over_S / (K * w_safe)

        # Update log states
        for idx_local, firm in enumerate(conglomerate):
            log_old = log_states_curr[firm]
            d = delta_over_state[idx_local]
            if not np.isfinite(d) or d <= -1.0:
                new_log_states[firm] = 0.0  # Reset to log(1)
                firms_to_exit.append(firm)
            else:
                new_log_states[firm] = log_old + np.log1p(d)

    firms_to_exit = np.array(firms_to_exit, dtype=np.int64)

    return new_log_states, cong_pools, firms_to_exit


# ============================================================================
# TEST SETUP
# ============================================================================

def create_test_scenario(scenario_name):
    """Create synthetic test data for different scenarios."""

    if scenario_name == "simple":
        # Simple scenario: 2 conglomerates, 3 firms each
        total_firms = 10
        markets = 5

        # Conglomerate structure
        cong_firms = np.full((5, markets), -1, dtype=np.int32)
        cong_firms[0, :3] = [0, 2, 4]  # Cong 0: firms 0,2,4
        cong_firms[1, :2] = [1, 3]      # Cong 1: firms 1,3

        cong_size = np.array([3, 2, 0, 0, 0], dtype=np.int16)
        cong_active = np.array([True, True, False, False, False])
        active_cong_ids = np.array([0, 1], dtype=np.int64)

        # Firm states and returns
        log_states_curr = np.array([0.5, 0.3, 0.6, 0.4, 0.7, 0.2, 0.1, 0.15, 0.25, 0.35])
        log_returns = np.array([0.05, 0.03, 0.04, 0.06, 0.02, 0.01, 0.005, 0.015, 0.025, 0.035])

    elif scenario_name == "extreme_values":
        # Test with extreme values (large/small states)
        total_firms = 8
        markets = 4

        cong_firms = np.full((4, markets), -1, dtype=np.int32)
        cong_firms[0, :2] = [0, 1]  # Small states
        cong_firms[1, :3] = [2, 3, 4]  # Large states

        cong_size = np.array([2, 3, 0, 0], dtype=np.int16)
        cong_active = np.array([True, True, False, False])
        active_cong_ids = np.array([0, 1], dtype=np.int64)

        log_states_curr = np.array([-5.0, -4.5, 5.0, 4.8, 5.2, 0.0, 0.0, 0.0])
        log_returns = np.array([0.1, 0.15, 0.05, 0.08, 0.06, 0.0, 0.0, 0.0])

    elif scenario_name == "exit_conditions":
        # Test exit conditions (negative deltas)
        total_firms = 6
        markets = 3

        cong_firms = np.full((3, markets), -1, dtype=np.int32)
        cong_firms[0, :2] = [0, 1]  # Will have exit

        cong_size = np.array([2, 0, 0], dtype=np.int16)
        cong_active = np.array([True, False, False])
        active_cong_ids = np.array([0], dtype=np.int64)

        log_states_curr = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        log_returns = np.array([-2.5, -2.0, 0.0, 0.0, 0.0, 0.0])  # Large negative returns

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    # Common parameters
    management_costs_lookup = np.array([0.0, 0.0, 0.001, 0.002, 0.003, 0.004])
    share = 0.5
    proportional = True
    lookback = 10
    step = 5

    return {
        'cong_firms': cong_firms,
        'cong_size': cong_size,
        'active_cong_ids': active_cong_ids,
        'log_returns': log_returns,
        'log_states_curr': log_states_curr,
        'management_costs_lookup': management_costs_lookup,
        'share': share,
        'proportional': proportional,
        'lookback': lookback,
        'step': step,
        'total_firms': total_firms
    }


def compare_results(python_results, numba_results, scenario_name):
    """Compare outputs from both implementations."""
    py_states, py_pools, py_exits = python_results
    nb_states, nb_pools, nb_exits = numba_results

    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")

    all_match = True

    # Compare new log states
    if np.allclose(py_states, nb_states, rtol=1e-10, atol=1e-12):
        max_diff = np.max(np.abs(py_states - nb_states))
        print(f"✅ Log states: MATCH (max diff: {max_diff:.2e})")
    else:
        max_diff = np.max(np.abs(py_states - nb_states))
        print(f"❌ Log states: MISMATCH (max diff: {max_diff:.2e})")
        diff_idx = np.argmax(np.abs(py_states - nb_states))
        print(f"   Worst at index {diff_idx}: Python={py_states[diff_idx]:.10f}, Numba={nb_states[diff_idx]:.10f}")
        all_match = False

    # Compare pool values
    if np.allclose(py_pools, nb_pools, rtol=1e-10, atol=1e-12):
        max_diff = np.max(np.abs(py_pools - nb_pools)) if len(py_pools) > 0 else 0.0
        print(f"✅ Pool values: MATCH (max diff: {max_diff:.2e})")
    else:
        max_diff = np.max(np.abs(py_pools - nb_pools))
        print(f"❌ Pool values: MISMATCH (max diff: {max_diff:.2e})")
        for i in range(len(py_pools)):
            print(f"   Pool {i}: Python={py_pools[i]:.10f}, Numba={nb_pools[i]:.10f}")
        all_match = False

    # Compare exits
    py_exits_sorted = np.sort(py_exits)
    nb_exits_sorted = np.sort(nb_exits)

    if np.array_equal(py_exits_sorted, nb_exits_sorted):
        print(f"✅ Exiting firms: MATCH ({len(py_exits)} firms)")
        if len(py_exits) > 0:
            print(f"   Firms: {py_exits_sorted}")
    else:
        print(f"❌ Exiting firms: MISMATCH")
        print(f"   Python: {py_exits_sorted}")
        print(f"   Numba: {nb_exits_sorted}")
        all_match = False

    return all_match


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ISOLATED POOLING LOGIC TEST: Python vs Numba")
    print("="*80)

    scenarios = ["simple", "extreme_values", "exit_conditions"]
    all_passed = True

    for scenario in scenarios:
        # Create test data
        test_data = create_test_scenario(scenario)

        # Remove total_firms for function calls (it's not a parameter)
        total_firms = test_data.pop('total_firms')

        # Run Python version
        python_results = process_conglomerate_pooling_python(**test_data)

        # Run Numba version
        numba_results = process_conglomerate_pooling_numba(**test_data)

        # Compare
        passed = compare_results(python_results, numba_results, scenario)
        all_passed = all_passed and passed

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Numba and Python implementations are equivalent!")
    else:
        print("❌ SOME TESTS FAILED - Implementations differ!")
    print("="*80 + "\n")
