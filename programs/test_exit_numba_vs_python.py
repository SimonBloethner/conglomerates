"""
Unit test: Compare Numba exit check vs Python exit check.
"""
import numpy as np
import numba as nb

# ============================================================================
# NUMBA VERSION
# ============================================================================

@nb.njit(cache=True)
def compute_exit_candidates_numba(
    firm_outside_log_profits,      # shape (lookback, total_firms)
    firm_log_states_buffer,        # shape (lookback+1, total_firms)
    exit_candidates,               # 1D array of firm IDs
    lookback,
    step
):
    """
    Numba-compiled exit evaluation logic (optimized version).

    Returns array of firm IDs that should exit (not a mask).
    """
    n = len(exit_candidates)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    lookback_plus_1 = firm_log_states_buffer.shape[0]
    start_idx = (step - lookback) % lookback_plus_1

    # Track which firms should exit
    exit_list = np.empty(n, dtype=np.int64)
    exit_count = 0

    for firm_id in exit_candidates:
        # Outside profit: arithmetic mean of log profits → log of geometric mean
        log_outside = 0.0
        for t in range(lookback):
            log_outside += firm_outside_log_profits[t, firm_id]
        log_outside /= lookback

        # Extract lookback+1 states and compute returns using direct modulo arithmetic
        # This eliminates array allocation - just compute indices on the fly
        log_inside = 0.0
        for i in range(lookback):
            curr_idx = (start_idx + i) % lookback_plus_1
            next_idx = (start_idx + i + 1) % lookback_plus_1
            curr_state = firm_log_states_buffer[curr_idx, firm_id]
            next_state = firm_log_states_buffer[next_idx, firm_id]
            log_inside += (next_state - curr_state)
        log_inside /= lookback

        if log_outside > log_inside:
            exit_list[exit_count] = firm_id
            exit_count += 1

    return exit_list[:exit_count]


# ============================================================================
# PYTHON VERSION
# ============================================================================

def compute_exit_candidates_python(
    firm_outside_log_profits,
    firm_log_states_buffer,
    exit_candidates,
    lookback,
    step
):
    """Python version using vectorization."""
    if len(exit_candidates) == 0:
        return np.zeros(0, dtype=bool)

    # Compute outside profits (vectorized)
    log_outside_profits = np.mean(firm_outside_log_profits[:, exit_candidates], axis=0)

    # Compute inside profits (manual loop for circular buffer)
    lookback_plus_1 = firm_log_states_buffer.shape[0]
    log_inside_profits = np.zeros(len(exit_candidates))

    for i, firm_id in enumerate(exit_candidates):
        log_inside_profit = 0.0
        for t in range(lookback):
            curr_step = step - lookback + t
            next_step = curr_step + 1

            curr_idx = curr_step % lookback_plus_1
            next_idx = next_step % lookback_plus_1

            log_return = firm_log_states_buffer[next_idx, firm_id] - firm_log_states_buffer[curr_idx, firm_id]
            log_inside_profit += log_return

        log_inside_profits[i] = log_inside_profit / lookback

    # Find which should exit
    exit_mask = log_outside_profits > log_inside_profits

    return exit_mask


# ============================================================================
# TESTS
# ============================================================================

def test_scenario(scenario_name):
    """Test exit logic with different scenarios."""

    lookback = 5
    total_firms = 10

    if scenario_name == "no_exits":
        # Inside profits better than outside
        step = 10

        firm_outside_log_profits = np.random.uniform(0.01, 0.03, (lookback, total_firms))
        firm_log_states_buffer = np.zeros((lookback + 1, total_firms))

        # Create increasing states (positive returns)
        for t in range(lookback + 1):
            firm_log_states_buffer[t] = np.arange(total_firms) * 0.1 + t * 0.05

        exit_candidates = np.array([1, 3, 5, 7])

    elif scenario_name == "some_exits":
        # Mixed: some should exit, some shouldn't
        step = 10

        firm_outside_log_profits = np.random.uniform(0.02, 0.08, (lookback, total_firms))
        firm_log_states_buffer = np.zeros((lookback + 1, total_firms))

        # Half have good inside returns, half have bad
        for t in range(lookback + 1):
            firm_log_states_buffer[t, :5] = t * 0.01  # Poor inside performance
            firm_log_states_buffer[t, 5:] = t * 0.10  # Good inside performance

        exit_candidates = np.array([1, 3, 6, 8])

    elif scenario_name == "all_exit":
        # Outside always better
        step = 10

        firm_outside_log_profits = np.random.uniform(0.1, 0.2, (lookback, total_firms))
        firm_log_states_buffer = np.zeros((lookback + 1, total_firms))

        # Declining states (negative returns)
        for t in range(lookback + 1):
            firm_log_states_buffer[t] = 10.0 - t * 0.5

        exit_candidates = np.array([0, 2, 4, 6, 8])

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    return {
        'firm_outside_log_profits': firm_outside_log_profits,
        'firm_log_states_buffer': firm_log_states_buffer,
        'exit_candidates': exit_candidates,
        'lookback': lookback,
        'step': step
    }


def compare_results(python_mask, numba_firm_ids, exit_candidates, scenario_name):
    """
    Compare exit results.

    python_mask: boolean array (one per candidate)
    numba_firm_ids: array of actual firm IDs that should exit
    exit_candidates: original candidate firm IDs
    """
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")

    # Convert Python mask to firm IDs for comparison
    python_firm_ids = exit_candidates[python_mask]

    # Sort both for comparison
    python_sorted = np.sort(python_firm_ids)
    numba_sorted = np.sort(numba_firm_ids)

    if np.array_equal(python_sorted, numba_sorted):
        n_exits = len(python_sorted)
        print(f"✅ Exit decisions: MATCH ({n_exits} firms exit)")
        if n_exits > 0:
            print(f"   Exiting firms: {python_sorted}")
        return True
    else:
        print(f"❌ Exit decisions: MISMATCH")
        print(f"   Python firms: {python_sorted}")
        print(f"   Numba firms:  {numba_sorted}")
        return False


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ISOLATED EXIT LOGIC TEST: Python vs Numba")
    print("="*80)

    scenarios = ["no_exits", "some_exits", "all_exit"]
    all_passed = True

    for scenario in scenarios:
        # Create test data
        test_data = test_scenario(scenario)
        exit_candidates = test_data['exit_candidates']

        # Run Python version (returns mask)
        python_mask = compute_exit_candidates_python(**test_data)

        # Run Numba version (returns firm IDs)
        numba_firm_ids = compute_exit_candidates_numba(**test_data)

        # Compare
        passed = compare_results(python_mask, numba_firm_ids, exit_candidates, scenario)
        all_passed = all_passed and passed

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Numba and Python exit logic are equivalent!")
    else:
        print("❌ SOME TESTS FAILED - Implementations differ!")
    print("="*80 + "\n")
