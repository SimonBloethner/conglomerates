"""
Unit tests to verify that log-space mathematical transformations
are equivalent to level-space operations.

This tests the MATH, not full simulation results.
"""

import numpy as np
from scipy.special import logsumexp, expm1


def test_solo_firm_update():
    """
    Test: Solo firm multiplicative update
    Level: s[t+1] = s[t] * r[t]
    Log:   L[t+1] = L[t] + log(r[t])
    """
    print("="*60)
    print("TEST 1: Solo Firm Update")
    print("="*60)

    # Level space
    s_t = 100.0
    r_t = 1.05  # 5% return
    s_t1_level = s_t * r_t

    # Log space
    L_t = np.log(s_t)
    log_r_t = np.log(r_t)
    L_t1 = L_t + log_r_t
    s_t1_log = np.exp(L_t1)

    print(f"Level space: s[t+1] = {s_t} * {r_t} = {s_t1_level}")
    print(f"Log space:   s[t+1] = exp({L_t:.6f} + {log_r_t:.6f}) = {s_t1_log}")
    print(f"Match: {np.isclose(s_t1_level, s_t1_log)}")
    assert np.isclose(s_t1_level, s_t1_log), "Solo firm update failed!"
    print("✅ PASS\n")


def test_gains_calculation():
    """
    Test: Individual gains calculation
    Level: g_i = s_i * r_i - s_i = s_i * (r_i - 1)
    Log:   g_i = exp(L_i) * expm1(log_r_i)
    """
    print("="*60)
    print("TEST 2: Gains Calculation")
    print("="*60)

    # Level space
    s_i = 100.0
    r_i = 1.08  # 8% return
    g_i_level = s_i * (r_i - 1)

    # Log space
    L_i = np.log(s_i)
    log_r_i = np.log(r_i)
    g_i_log = np.exp(L_i) * expm1(log_r_i)

    print(f"Level space: g_i = {s_i} * ({r_i} - 1) = {g_i_level}")
    print(f"Log space:   g_i = exp({L_i:.6f}) * expm1({log_r_i:.6f}) = {g_i_log}")
    print(f"Match: {np.isclose(g_i_level, g_i_log)}")
    assert np.isclose(g_i_level, g_i_log), "Gains calculation failed!"
    print("✅ PASS\n")


def test_weighted_average_gains():
    """
    Test: Weighted average of gains for pooling
    Level: avg_gain = sum(s_i * (r_i - 1)) / sum(s_i)
    Log:   avg_gain = sum(w_i * (r_i - 1)) where w_i = exp(L_i - log_S)
    """
    print("="*60)
    print("TEST 3: Weighted Average Gains")
    print("="*60)

    # Level space
    states = np.array([100.0, 200.0, 150.0])
    returns = np.array([1.05, 1.08, 1.03])
    gains = states * (returns - 1)
    avg_gain_level = np.sum(gains) / np.sum(states)

    # Log space
    log_states = np.log(states)
    log_returns = np.log(returns)
    log_S = logsumexp(log_states)
    weights = np.exp(log_states - log_S)
    r_minus1 = expm1(log_returns)
    avg_gain_log = np.sum(weights * r_minus1)

    print(f"Level space: sum(s_i*(r_i-1)) / sum(s_i) = {avg_gain_level:.6f}")
    print(f"Log space:   sum(w_i*(r_i-1)) = {avg_gain_log:.6f}")
    print(f"Weights sum to 1: {np.sum(weights):.10f}")
    print(f"Match: {np.isclose(avg_gain_level, avg_gain_log)}")
    assert np.isclose(avg_gain_level, avg_gain_log), "Weighted average gains failed!"
    print("✅ PASS\n")


def test_proportional_pool():
    """
    Test: Pool calculation (proportional cost)
    Level: pool = share * sum(g_i) * (1 - C/sum(s_i))
    Log:   pool = share * S * avg_gain * (1 - C/S)
          where S = exp(log_S), avg_gain from weights
    """
    print("="*60)
    print("TEST 4: Proportional Pool Calculation")
    print("="*60)

    # Setup
    states = np.array([100.0, 200.0, 150.0])
    returns = np.array([1.05, 1.08, 1.03])
    share = 0.5
    C = 0.001  # Management cost

    # Level space
    gains = states * (returns - 1)
    sum_gains = np.sum(gains)
    sum_states = np.sum(states)
    pool_level = share * sum_gains * (1 - C / sum_states)

    # Log space approach
    log_states = np.log(states)
    log_returns = np.log(returns)
    log_S = logsumexp(log_states)
    S = np.exp(log_S)
    weights = np.exp(log_states - log_S)
    r_minus1 = expm1(log_returns)
    avg_gain_weighted = np.sum(weights * r_minus1)
    mgmt_over_S = C / S
    cost_factor = 1.0 - mgmt_over_S
    pool_log = share * S * avg_gain_weighted * cost_factor

    print(f"Level space: pool = {pool_level:.6f}")
    print(f"Log space:   pool = {pool_log:.6f}")
    print(f"Match: {np.isclose(pool_level, pool_log)}")
    assert np.isclose(pool_level, pool_log), "Proportional pool failed!"
    print("✅ PASS\n")


def test_non_proportional_pool():
    """
    Test: Pool calculation (non-proportional cost)
    Level: pool = share * sum(g_i) - C * sum(s_i)
    Log:   pool = share * S * avg_gain - C * S
    """
    print("="*60)
    print("TEST 5: Non-Proportional Pool Calculation")
    print("="*60)

    # Setup
    states = np.array([100.0, 200.0, 150.0])
    returns = np.array([1.05, 1.08, 1.03])
    share = 0.5
    C = 0.001  # Management cost

    # Level space
    gains = states * (returns - 1)
    sum_gains = np.sum(gains)
    sum_states = np.sum(states)
    pool_level = share * sum_gains - C * sum_states

    # Log space
    log_states = np.log(states)
    log_returns = np.log(returns)
    log_S = logsumexp(log_states)
    S = np.exp(log_S)
    weights = np.exp(log_states - log_S)
    r_minus1 = expm1(log_returns)
    avg_gain_weighted = np.sum(weights * r_minus1)
    pool_log = share * S * avg_gain_weighted - C * S

    print(f"Level space: pool = {pool_level:.6f}")
    print(f"Log space:   pool = {pool_log:.6f}")
    print(f"Match: {np.isclose(pool_level, pool_log)}")
    assert np.isclose(pool_level, pool_log), "Non-proportional pool failed!"
    print("✅ PASS\n")


def test_conglomerate_state_update():
    """
    Test: Conglomerate firm state update
    Level: s_i[t+1] = s_i[t] + (1-share)*g_i + d_i
                    = s_i[t] * (1 + (1-share)*(r_i-1) + d_i/s_i)
    Log:   L_i[t+1] = L_i[t] + log1p((1-share)*(r_i-1) + d_i/s_i)
    """
    print("="*60)
    print("TEST 6: Conglomerate State Update")
    print("="*60)

    # Setup
    s_i = 100.0
    r_i = 1.05
    share = 0.5
    pool_total = 20.0
    n_firms = 3
    d_i = pool_total / n_firms  # Equal distribution

    # Level space
    g_i = s_i * (r_i - 1)
    s_i_new_level = s_i + (1 - share) * g_i + d_i

    # Log space approach
    L_i = np.log(s_i)
    log_r_i = np.log(r_i)
    r_minus1 = expm1(log_r_i)
    delta_over_state = (1 - share) * r_minus1 + d_i / s_i
    L_i_new = L_i + np.log1p(delta_over_state)
    s_i_new_log = np.exp(L_i_new)

    print(f"Level space: s[t+1] = {s_i} + {(1-share)*g_i:.4f} + {d_i:.4f} = {s_i_new_level:.4f}")
    print(f"Log space:   s[t+1] = exp({L_i:.6f} + log1p({delta_over_state:.6f})) = {s_i_new_log:.4f}")
    print(f"Match: {np.isclose(s_i_new_level, s_i_new_log)}")
    assert np.isclose(s_i_new_level, s_i_new_log), "Conglomerate update failed!"
    print("✅ PASS\n")


def test_full_conglomerate_cycle():
    """
    Test: Full conglomerate cycle combining all operations
    """
    print("="*60)
    print("TEST 7: Full Conglomerate Cycle")
    print("="*60)

    # Setup
    states = np.array([100.0, 200.0, 150.0])
    returns = np.array([1.05, 1.08, 1.03])
    share = 0.5
    C = 0.001
    proportional = True

    # LEVEL SPACE
    gains = states * (returns - 1)
    sum_states = np.sum(states)
    sum_gains = np.sum(gains)

    if proportional:
        pool = share * sum_gains * (1 - C / sum_states)
    else:
        pool = share * sum_gains - C * sum_states

    n_firms = len(states)
    distributions = np.full(n_firms, pool / n_firms)
    new_states_level = states + (1 - share) * gains + distributions

    # LOG SPACE
    log_states = np.log(states)
    log_returns = np.log(returns)
    log_S = logsumexp(log_states)
    S = np.exp(log_S)
    weights = np.exp(log_states - log_S)
    r_minus1 = expm1(log_returns)
    avg_gain = np.sum(weights * r_minus1)

    if proportional:
        mgmt_over_S = C / S
        cost_factor = 1.0 - mgmt_over_S
        pool_over_S = share * avg_gain * cost_factor
    else:
        pool_over_S = share * avg_gain - C

    # Distribution per firm relative to its weight
    weights_safe = np.maximum(weights, 1e-300)
    delta_over_state = (1 - share) * r_minus1 + pool_over_S / (n_firms * weights_safe)

    new_log_states = log_states + np.log1p(delta_over_state)
    new_states_log = np.exp(new_log_states)

    print(f"Level space new states: {new_states_level}")
    print(f"Log space new states:   {new_states_log}")
    print(f"Max difference: {np.max(np.abs(new_states_level - new_states_log)):.2e}")
    print(f"Match: {np.allclose(new_states_level, new_states_log)}")
    assert np.allclose(new_states_level, new_states_log), "Full cycle failed!"
    print("✅ PASS\n")


def test_geometric_mean_exit_check():
    """
    Test: Exit decision using geometric means
    Level: outside_GM = prod(r_i)^(1/n), inside_GM = prod(s[t]/s[t-1])^(1/n)
    Log:   outside_GM = exp(mean(log(r_i))), inside_GM = exp(mean(diff(log(s))))
    """
    print("="*60)
    print("TEST 8: Geometric Mean Exit Check")
    print("="*60)

    # Setup
    outside_returns = np.array([1.05, 1.08, 1.03, 1.06, 1.07])
    inside_states = np.array([100.0, 105.0, 112.0, 115.0, 121.0, 127.0])

    # Level space
    outside_GM_level = np.prod(outside_returns) ** (1 / len(outside_returns))
    inside_returns = inside_states[1:] / inside_states[:-1]
    inside_GM_level = np.prod(inside_returns) ** (1 / len(inside_returns))

    # Log space
    log_outside_returns = np.log(outside_returns)
    outside_GM_log = np.exp(np.mean(log_outside_returns))

    log_inside_states = np.log(inside_states)
    inside_log_returns = np.diff(log_inside_states)
    inside_GM_log = np.exp(np.mean(inside_log_returns))

    print(f"Level space - Outside GM: {outside_GM_level:.8f}")
    print(f"Log space   - Outside GM: {outside_GM_log:.8f}")
    print(f"Match: {np.isclose(outside_GM_level, outside_GM_log)}")
    assert np.isclose(outside_GM_level, outside_GM_log), "Outside GM failed!"

    print(f"Level space - Inside GM: {inside_GM_level:.8f}")
    print(f"Log space   - Inside GM: {inside_GM_log:.8f}")
    print(f"Match: {np.isclose(inside_GM_level, inside_GM_log)}")
    assert np.isclose(inside_GM_level, inside_GM_log), "Inside GM failed!"
    print("✅ PASS\n")


def test_market_share_calculation():
    """
    Test: Market share calculation
    Level: share_i = s_i / sum(s_j)
    Log:   share_i = exp(L_i - logsumexp(L_j))
    """
    print("="*60)
    print("TEST 9: Market Share Calculation")
    print("="*60)

    # Setup
    states = np.array([100.0, 250.0, 150.0, 300.0])

    # Level space
    total = np.sum(states)
    shares_level = states / total

    # Log space
    log_states = np.log(states)
    log_total = logsumexp(log_states)
    shares_log = np.exp(log_states - log_total)

    print(f"Level space shares: {shares_level}")
    print(f"Log space shares:   {shares_log}")
    print(f"Shares sum to 1 (level): {np.sum(shares_level):.10f}")
    print(f"Shares sum to 1 (log):   {np.sum(shares_log):.10f}")
    print(f"Match: {np.allclose(shares_level, shares_log)}")
    assert np.allclose(shares_level, shares_log), "Market share failed!"
    print("✅ PASS\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LOG-SPACE MATHEMATICAL TRANSFORMATION TESTS")
    print("="*60 + "\n")

    try:
        test_solo_firm_update()
        test_gains_calculation()
        test_weighted_average_gains()
        test_proportional_pool()
        test_non_proportional_pool()
        test_conglomerate_state_update()
        test_full_conglomerate_cycle()
        test_geometric_mean_exit_check()
        test_market_share_calculation()

        print("="*60)
        print("✅ ALL MATHEMATICAL TESTS PASSED!")
        print("="*60)
        print("\nConclusion: Log-space transformations are mathematically")
        print("equivalent to level-space operations.")

    except AssertionError as e:
        print("="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        exit(1)
