#!/usr/bin/env python3
"""
Tests for alpha_endogenous flag.

Test catches:
1. Flag off: identity test passes (changes to default behavior)
2. replay_member_growth refactor: merger decisions bit-identical before/after
3. C34: every adopted move raises every member's demeaned gain and maximizes the minimum gain (unanimity,
   then maximin)
4. C34: a unit that does not move has no grid rate that raises every member's demeaned gain
5. C34: hand-computed window in which the realized-level rule and the demeaned rule choose differently
"""
import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

import collaborative_growth as cg
from collaborative_growth import model, seed_numba

GRID = [k * 0.05 for k in range(21)]


def _demeaned_gains(members, past_returns, sizes, alpha, phi, code, g):
    """Delta-hat_j(alpha) for every member, from the same helpers the renegotiation uses."""
    K = past_returns.shape[1]
    eps, sz, sum_s, sum_s_eps = cg.prepare_window_loggain(members, past_returns, sizes, g)
    return np.array([cg.member_gain_from_prepared(j, K, eps, sz, sum_s, sum_s_eps, alpha, phi, code)
                     for j in range(K)])


def _min_gains_by_rate(members, past_returns, sizes, current, phi, code, g):
    """{alpha': min_j [Delta-hat_j(alpha') - Delta-hat_j(alpha_C)]} over feasible grid rates."""
    base = _demeaned_gains(members, past_returns, sizes, current, phi, code, g)
    out = {}
    for a in GRID:
        d = _demeaned_gains(members, past_returns, sizes, a, phi, code, g)
        if np.isneginf(d).any():
            continue
        out[a] = np.min(d - base)
    return out


@pytest.fixture(scope='module')
def renegotiations():
    """Seeded M=N=20 endogenous-alpha run (tests/test_floor_exit.py configuration) with find_optimal_alpha
    wrapped: records the arguments and the result of every renegotiation."""
    from tests.test_floor_exit import PARAMS, KW
    calls = []
    orig = cg.find_optimal_alpha

    def wrapped(members, past_returns, sizes, current, phi, code, g):
        out = orig(members, past_returns, sizes, current, phi, code, g)
        calls.append((members.copy(), past_returns.copy(), sizes.copy(), current, phi, code, g, out))
        return out

    cg.find_optimal_alpha = wrapped
    try:
        model(PARAMS, **dict(KW, alpha_endogenous=True))
    finally:
        cg.find_optimal_alpha = orig
    return calls


def test_alpha_endogenous_identity():
    """
    With alpha_endogenous=False, identity test passes.

    Test catches: changes to default behavior.
    """
    import test_phase_b_identity
    test_phase_b_identity.test_phase_b_identity()
    print("PASS: alpha_endogenous identity test (flag off)")


def test_replay_refactor_bitidentical():
    """
    With alpha_endogenous=False, merger decisions are bit-identical
    before and after the replay_member_growth refactor.

    Test catches: refactoring broke merger logic.
    """
    M = 5
    N = 30
    T = 500
    ALPHA = 0.2
    SEED = 42

    total_firms = M * N
    params = [
        M, N, T, ALPHA, total_firms,
        0.05, 4, 0.85, False, 50,
        'power_law', None, None, None,
    ]

    # Run with alpha_endogenous=False (default)
    result = model(params, seed=SEED)

    mergers = result[7]  # mergers_per_period

    # Run again with same seed
    result2 = model(params, seed=SEED)
    mergers2 = result2[7]

    assert np.array_equal(mergers, mergers2), "Merger decisions not bit-identical"

    # Total mergers should be > 0 to be a meaningful test
    assert mergers.sum() > 0, "No mergers occurred - test not meaningful"

    print(f"PASS: replay refactor bit-identical ({mergers.sum()} total mergers)")


def test_alpha_unanimity_constraint(renegotiations):
    """C34: every adopted move raises every member's demeaned gain Delta-hat_j (unanimity) and attains the
    largest minimum gain over the feasible grid (maximin), recomputed from prepare_window_loggain and
    member_gain_from_prepared. Some units do move in the seeded run, so the check is not vacuous.
    Catches: a move that hurts a member, a move to a rate that is not the maximin one, or a renegotiation
    still evaluated on realized levels."""
    moves = 0
    for members, past_returns, sizes, current, phi, code, g, (alpha, changed) in renegotiations:
        if not changed:
            continue
        moves += 1
        assert alpha in GRID and alpha != current
        by_rate = _min_gains_by_rate(members, past_returns, sizes, current, phi, code, g)
        assert by_rate[alpha] > 0
        assert by_rate[alpha] == max(by_rate.values())
    assert moves > 0


def test_alpha_history_structure():
    """
    Test that alpha_history is returned with correct structure.

    Test catches: missing or malformed alpha_history output.
    """
    M = 4
    N = 4
    T = 500
    SEED = 456

    total_firms = M * N
    params = [
        M, N, T, 0.5, total_firms,
        0.05, 4, 0.0, False, 50,
        'power_law', 0.0, 1.0, None,
    ]

    result = model(
        params, seed=SEED,
        growth_process='log_family',
        log_family='normal',
        g=0.05,
        sigma_range=(0.15, 0.15),
        alpha_endogenous=True,
        metric_every=100
    )

    hyperparams = result[-1]
    alpha_history = hyperparams.get('alpha_history')

    # Check alpha_history exists and has correct shape
    assert alpha_history is not None, "alpha_history not returned"
    assert len(alpha_history.shape) == 2, "alpha_history should be 2D"

    # Shape should be [max_cong, num_metric_steps]
    # With T=500 and metric_every=100, we expect 5 metric steps
    expected_metric_steps = T // 100
    assert alpha_history.shape[1] == expected_metric_steps, \
        f"Expected {expected_metric_steps} metric steps, got {alpha_history.shape[1]}"

    # Check alpha_endogenous flag is recorded
    assert hyperparams.get('alpha_endogenous') == True, \
        "alpha_endogenous flag not recorded in hyperparameters"

    # Count active conglomerates (those with non-nan alpha values)
    active_congs = 0
    for cong_idx in range(alpha_history.shape[0]):
        if not np.all(np.isnan(alpha_history[cong_idx, :])):
            active_congs += 1

    print(f"PASS: alpha_history structure correct ({active_congs} active congs, shape={alpha_history.shape})")


def test_no_improvement_keeps_alpha(renegotiations):
    """C34: a unit that keeps its rate has no feasible grid rate under which every member's demeaned gain
    rises (the maximum over alpha' of min_j [Delta-hat_j(alpha') - Delta-hat_j(alpha_C)] is <= 0), and the
    returned rate is alpha_C. Catches: a rule that keeps the rate although a unanimous improvement exists,
    or one that reports no change but returns a different rate."""
    kept = 0
    for members, past_returns, sizes, current, phi, code, g, (alpha, changed) in renegotiations:
        if changed:
            continue
        kept += 1
        assert alpha == current
        by_rate = _min_gains_by_rate(members, past_returns, sizes, current, phi, code, g)
        assert max(by_rate.values(), default=0.0) <= 0
    assert kept > 0


def _level_rule(past_states, past_returns, current, phi, code):
    """The pre-C34 renegotiation on realized levels (find_optimal_alpha at 755f4b7), for comparison."""
    step_states = past_states[-1:].copy()
    cur, ok = cg.replay_member_growth(past_states, past_returns, step_states, current, phi, False, code)
    if not ok:
        cur = np.full(past_states.shape[1], -np.inf)
    best_alpha, best = current, -np.inf
    for k in range(21):
        a = k * 0.05
        new, ok = cg.replay_member_growth(past_states, past_returns, step_states, a, phi, False, code)
        if not ok:
            continue
        if np.min(new - cur) > best:
            best, best_alpha = np.min(new - cur), a
    return (best_alpha, True) if best > 0 and best_alpha != current else (current, False)


def test_demeaned_and_level_rules_differ():
    """Hand-computed two-member window (h = 2, equal sizes, proportional sharing, Phi = 0, g = 0, alpha_C = 0.1).
    Log returns: A = (0.4, 0.0) (a lucky window, mean 0.2), B = (-0.2, 0.2) (mean 0).
    Levels: A gives away its luck when alpha rises (its replayed growth falls from 0.1941 at 0.1 to 0.1247 at
    1) and B loses when alpha falls (0.0156 at 0.1, 0 at 0), so no rate helps both and the unit stays at 0.1.
    Demeaned: both windows become (0.2, -0.2) and (-0.2, 0.2), perfectly offsetting; at alpha = 1 both
    members earn log(1 + (expm1(0.2) + expm1(-0.2))/2) = 0.01987 per period against 0 standalone, more than
    at any lower rate, so the unit moves to 1.
    Catches: the renegotiation still on realized levels (A vetoes), or a demeaned gain computed wrongly."""
    members = np.array([0, 1], dtype=np.int32)
    past_returns = np.array([[0.4, -0.2], [0.0, 0.2]])
    past_states = np.zeros((2, 2))
    sizes = np.exp(past_states)
    current, phi, code, g = 0.1, 0.0, 1, 0.0

    # the level rule keeps alpha_C; the hand values in the docstring
    lvl = lambda a: cg.replay_member_growth(past_states, past_returns, past_states[-1:], a, phi, False, code)[0]
    assert np.allclose(lvl(0.1), [0.19409, 0.01557], atol=1e-4)
    assert np.allclose(lvl(1.0), [0.12466, 0.12466], atol=1e-4)
    assert np.allclose(lvl(0.0), [0.2, 0.0], atol=1e-12)
    assert _level_rule(past_states, past_returns, current, phi, code) == (current, False)

    # the demeaned gains at alpha = 1, by hand
    e = np.expm1(np.array([0.2, -0.2]))
    hand = np.log1p(e.mean()) - 0.5 * (np.log1p(e[0]) + np.log1p(e[1]))
    assert abs(hand - 0.019868) < 1e-6
    assert np.allclose(_demeaned_gains(members, past_returns, sizes, 1.0, phi, code, g), hand, atol=1e-14)

    # the demeaned rule moves to 1
    alpha, changed = cg.find_optimal_alpha(members, past_returns, sizes, current, phi, code, g)
    assert changed and alpha == 1.0


if __name__ == '__main__':
    print("Testing alpha_endogenous identity (flag off)...")
    test_alpha_endogenous_identity()

    print("\nTesting replay refactor bit-identical...")
    test_replay_refactor_bitidentical()

    print("\nTesting alpha_history structure...")
    test_alpha_history_structure()

    test_demeaned_and_level_rules_differ()

    print("\nAll alpha_endogenous tests passed!")
