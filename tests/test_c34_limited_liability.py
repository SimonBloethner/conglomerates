"""C34a: limited liability. The shortfall of a member who cannot pay its assigned loss is charged to the
other members of its unit (in proportion to their post-pooling capital), not to its market.

Hand-computed cases call process_conglomerate_pooling_numba directly (Phi given by the cost lookup, cost
charged as a growth drag as in the pilot). The conservation tests wrap the function in seeded model runs and
check, for every active unit in every period,
    |sum_j s_j,t+1 - [sum_j s_j (1 + r_j) - Phi(K) S]| < 1e-9 S,
with capped members contributing their capped size.
"""
import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW

CAP = -1.0 + 1e-12
KEEP = 1.0 + CAP  # growth factor of a capped member (~1e-12; 0.99998894e-12 in floating point)


def _pool(sizes, rel_returns, alpha, phi, sharing_rule_code):
    """One unit holding firms 0..K-1; returns (new sizes, pool_ruin, pool_shortfall, pool_wipeout)."""
    K = len(sizes)
    cong_firms = np.arange(K, dtype=np.int32).reshape(1, K)
    cong_size = np.array([K], dtype=np.int16)
    active = np.array([0], dtype=np.int64)
    log_returns = np.log1p(np.asarray(rel_returns, dtype=np.float64))
    log_states = np.log(np.asarray(sizes, dtype=np.float64))
    costs = np.zeros(K + 1)
    costs[K] = phi
    out = cg.process_conglomerate_pooling_numba(cong_firms, cong_size, active, log_returns, log_states, costs,
                                                alpha, False, 50, 0, sharing_rule_code,
                                                np.full(1, alpha), False)
    new_log_states, _, ruin, shortfall, wipeout = out
    return np.exp(new_log_states), ruin, shortfall, wipeout


def test_two_member_shortfall_charged_to_other_member():
    """Sizes (100, 0.5), returns (-20%, 0), alpha 0.3, equal sharing, Phi 0. Pool -6, so each member is
    assigned -3: the large member's intended capital is 100 - 14 - 3 = 83, the small member's 0.5 - 3 = -2.5.
    The small member is capped (keeps 0.5 KEEP ~ 0.5e-12) and its shortfall 2.5 + 0.5 KEEP is charged to the large one,
    which ends at 80.5 - 0.5 KEEP = sum_j s_j (1 + r_j) minus what the capped member keeps.
    Catches: the C33 behaviour (large member stays at 83, the 2.5 is created out of nothing and later taken
    from the small member's market by the renormalization)."""
    new, ruin, sf, wipe = _pool([100.0, 0.5], [-0.2, 0.0], 0.3, 0.0, 0)
    assert ruin == 1 and wipe == 0
    assert abs(sf - (2.5 + 0.5 * KEEP)) < 1e-12
    assert abs(new[1] - 0.5 * KEEP) < 1e-12 * 0.5 * KEEP
    assert abs(new[0] - (80.5 - 0.5 * KEEP)) < 1e-10


def test_three_member_shortfall_split_by_capital():
    """Sizes (100, 50, 0.5), returns (-20%, 0, 0), alpha 0.3, equal sharing, Phi 0. Each member is assigned
    -2 from the pool: intended capital 84, 48, -1.5. The third member is capped; its shortfall 1.5 + 0.5 KEEP
    is split 84:48 between the other two.
    Catches: a shortfall split equally or by initial size instead of by post-pooling capital, or charged to
    only one member."""
    new, ruin, sf, wipe = _pool([100.0, 50.0, 0.5], [-0.2, 0.0, 0.0], 0.3, 0.0, 0)
    sf_hand = 1.5 + 0.5 * KEEP
    assert ruin == 1 and wipe == 0
    assert abs(sf - sf_hand) < 1e-12
    assert abs(new[0] - (84.0 - sf_hand * 84.0 / 132.0)) < 1e-10
    assert abs(new[1] - (48.0 - sf_hand * 48.0 / 132.0)) < 1e-10
    assert abs(new[2] - 0.5 * KEEP) < 1e-12 * 0.5 * KEEP
    assert abs(new.sum() - 130.5) < 1e-10


def test_cascade():
    """Equal sharing, alpha 1, zero returns, Phi chosen so that every member is assigned D = -(10 - 1e-8)
    (pool -Phi S, split equally). Sizes (5, 10, 15.000997):
      round 1: member 0 (5 + D < 0) is capped; shortfall SF1 = KEEP*5 - (5 + D);
               factor f = 1 - SF1 / (c1 + c2) with c1 = 10 + D = 1e-8, c2 = 15.000997 + D;
      round 2: member 1 holds c1 f ~ 2e-12 < 1e-12 * 10, so it is capped; SF2 = KEEP*10 - c1 f, charged to member 2.
    Member 2 ends at the unit identity S - Phi S minus what the two capped members keep.
    Catches: a single pass (member 1 left below the cap), or a cascade that loses or double-charges capital."""
    s = np.array([5.0, 10.0, 15.000997])
    S = s.sum()
    D = -(10.0 - 1e-8)
    phi = -3.0 * D / S
    new, ruin, sf, wipe = _pool(s, [0.0, 0.0, 0.0], 1.0, phi, 0)
    # hand computation (same formulas as the docstring)
    c = s + (-phi * S) / 3.0
    sf1 = KEEP * s[0] - c[0]
    f = 1.0 - sf1 / (c[1] + c[2])
    assert c[1] * f < KEEP * s[1]  # the case really cascades
    sf2 = KEEP * s[1] - c[1] * f
    assert ruin == 2 and wipe == 0
    assert abs(sf - (sf1 + sf2)) < 1e-12
    assert abs(new[0] - KEEP * s[0]) < 1e-12 * KEEP * s[0]
    assert abs(new[1] - KEEP * s[1]) < 1e-12 * KEEP * s[1]
    expected_2 = S - phi * S - KEEP * (s[0] + s[1])
    assert abs(new[2] - expected_2) < 1e-9 * S
    assert expected_2 > 0


def test_wipeout_counted():
    """Equal sharing, Phi = 1.2: the unit's intended capital S - 1.2 S is negative, every member is capped and
    the unit counts as one wipeout. Catches: an infinite loop, a negative capital, or an uncounted wipeout."""
    new, ruin, sf, wipe = _pool([5.0, 10.0, 15.0], [0.0, 0.0, 0.0], 1.0, 1.2, 0)
    assert wipe == 1 and ruin == 3
    assert np.allclose(new, KEEP * np.array([5.0, 10.0, 15.0]), rtol=1e-9, atol=0)


def test_proportional_sharing_cap():
    """Proportional sharing, sizes (1, 1), returns (-99%, 0), alpha 0.1, Phi 0.2: member 0's factor is
    0.9 (-0.99) + 0.1 (-0.495) - 0.2 = -1.1405, so it is capped and its shortfall 0.1405 (+KEEP) is charged
    to member 1, which ends at 0.01 + 1 - 0.4 = 0.61 (the identity) less KEEP.
    Catches: the limited-liability branch applied only under equal sharing."""
    new, ruin, sf, wipe = _pool([1.0, 1.0], [-0.99, 0.0], 0.1, 0.2, 1)
    assert ruin == 1 and wipe == 0
    assert abs(sf - (0.1405 + KEEP)) < 1e-12
    assert abs(new[1] - (0.61 - KEEP)) < 1e-12


def test_uncapped_unit_unchanged():
    """A unit with no capped member takes the C33 path, log_old + log1p(delta), with no shortfall. (Bit identity
    of whole runs without capped members is asserted in tests/test_c31_identity.py::test_proportional_unchanged.)
    Catches: the redistribution branch firing, or altering ordinary updates."""
    sizes = np.array([1.3, 0.7, 2.1])
    r = np.array([0.05, -0.1, 0.02])
    alpha, phi = 0.3, 0.004
    for code in (0, 1):
        new, ruin, sf, wipe = _pool(sizes, r, alpha, phi, code)
        log_st = np.log(sizes)
        log_S = np.log(np.exp(log_st - log_st.max()).sum()) + log_st.max()
        w = np.exp(log_st - log_S)
        rm1 = np.expm1(np.log1p(r))
        pool = alpha * (w * rm1).sum() - phi
        delta = (1 - alpha) * rm1 + (pool / (3 * w) if code == 0 else pool)
        assert ruin == 0 and sf == 0.0 and wipe == 0
        assert np.allclose(new, sizes * (1 + delta), rtol=1e-13, atol=0)


def _conservation_run(**overrides):
    """Seeded M=N=20, T=600 run (tests/test_floor_exit.py configuration) with the pooling function wrapped:
    records the largest relative conservation error over all active units and periods."""
    orig = cg.process_conglomerate_pooling_numba
    rec = {'max_rel_err': 0.0, 'units': 0, 'capped_units': 0, 'shortfall': 0.0}

    def wrapped(*a):
        out = orig(*a)
        cong_firms, cong_size, active_ids, log_returns, log_states, costs = a[:6]
        new_log_states, _, ruin, shortfall, wipeout = out
        assert wipeout == 0
        for cid in active_ids:
            members = cong_firms[cid, :cong_size[cid]]
            s = np.exp(log_states[members])
            S = s.sum()
            identity = (s * (1.0 + np.expm1(log_returns[members]))).sum() - costs[len(members)] * S
            after = np.exp(new_log_states[members]).sum()
            rec['max_rel_err'] = max(rec['max_rel_err'], abs(after - identity) / S)
            rec['units'] += 1
        if ruin > 0:
            rec['capped_units'] += 1
            rec['shortfall'] += shortfall
        return out

    cg.process_conglomerate_pooling_numba = wrapped
    try:
        hp = cg.model(PARAMS, **dict(KW, **overrides))[-1]
    finally:
        cg.process_conglomerate_pooling_numba = orig
    return hp, rec


@pytest.mark.parametrize('sharing_rule', ['equal', 'proportional'])
def test_conservation_in_model_run(sharing_rule):
    """Every active unit in every period conserves sum_j s_j (1 + r_j) - Phi(K) S to 1e-9 S. The equal-split
    run has capped members (so the redistribution is exercised); the proportional run guards the main path.
    Catches: a shortfall that is dropped (C33: the unit then holds more than the identity) or over-charged."""
    hp, rec = _conservation_run(sharing_rule=sharing_rule)
    assert rec['units'] > 0
    assert rec['max_rel_err'] < 1e-9, rec
    if sharing_rule == 'equal':
        assert rec['capped_units'] > 0 and rec['shortfall'] > 0
        assert abs(hp['pool_shortfall_per_period'].sum() - rec['shortfall']) < 1e-9 * rec['shortfall']
    assert hp['summary']['pool_wipeout_total'] == 0
    assert hp['summary']['pool_shortfall_per_period'] == pytest.approx(
        np.mean(hp['pool_shortfall_per_period'][KW['burn_in']:]))
