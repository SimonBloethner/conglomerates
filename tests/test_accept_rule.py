"""C42 Task 2: decision_rule='accept' (code 4), the survival-feasible ceiling. A control without knowledge: every
feasible proposal (at most one firm per market in the merged set) is accepted, no member gain is computed, and the
exit review never produces a voluntary exit. Floor exits, pooling, cost and the floor are unchanged."""
import os

import numpy as np
import pandas as pd
import pytest

import collaborative_growth as cg
from tests.test_c34_limited_liability import _conservation_run
from tests.test_c38_acceptance_by_K_w import kernel
from tests.test_floor_exit import PARAMS, KW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCEPT = 4

# Three markets, two firms each (firm f in market f // 2): unit {0, 2} (markets 0, 1), standalones 1 (market 0),
# 3 (market 1), 4 and 5 (market 2). Feasible: solo+solo across markets, unit {0, 2} with 4 or 5. Infeasible: 1 or 3
# with the unit (it occupies their market), 4 with 5 (same market).
LOG_SIZES = np.log([1.0, 0.4, 2.0, 1.5, 0.8, 3.0])
UNITS = [(0, 2)]


def feasible(units_of, home, a, b):
    """Brute-force feasibility of merging the owners of firms a and b: no market twice."""
    fa = units_of.get(a, (a,))
    fb = units_of.get(b, (b,))
    if set(fa) & set(fb):
        return False
    markets = [home[f] for f in fa + fb]
    return len(markets) == len(set(markets))


def test_accept_on_fixed_economy_merges_every_feasible_proposal():
    """(a) On the fixed three-market economy (the state is restored after every period), every proposal that
    reaches the test is merged under accept, while the moment rule refuses some on the same economy. (The two rules'
    proposal counts differ: a merger within a period changes what later initiators in that period can propose.)"""
    a4, props4, mergs4 = kernel(LOG_SIZES, units=UNITS, N=2, rule=ACCEPT, steps=200)
    a2, props2, mergs2 = kernel(LOG_SIZES, units=UNITS, N=2, rule=2, steps=200)
    assert props4.sum() > 0 and np.array_equal(mergs4, props4)
    assert mergs2.sum() < props2.sum()
    # the composition counters stay on: proposals and mergers by K, w and type are recorded and equal
    assert np.array_equal(a4['pt'], a4['mt']) and a4['pt'].sum() == props4.sum()
    assert a4['pt'][2, :, 0].sum() > 0 and a4['pt'][3, :, 1].sum() > 0
    # no binding gain is computed under accept
    assert a4['n'].sum() == 0 and a4['s1'].sum() == 0


def _evolving_accept(log_sizes, units, M, N, steps, seed):
    """Run merger_kernel_numba under accept for `steps` periods on an evolving state (mergers persist); after every
    period check the occupancy invariant and record (proposals, mergers)."""
    F = len(log_sizes)
    window = 10
    buf = np.tile(np.asarray(log_sizes, float), (window + 1, 1))
    ret = np.random.default_rng(seed).normal(0.055, 0.05, (window + 1, F))
    MAXU = F
    firm_unit = np.full(F, -1, np.int32)
    unit_members = np.full((MAXU, M), -1, np.int32)
    unit_size = np.zeros(MAXU, np.int16)
    unit_active = np.zeros(MAXU, bool)
    occ = np.zeros((MAXU, M), bool)
    for u, members in enumerate(units):
        unit_active[u] = True
        unit_size[u] = len(members)
        unit_members[u, :len(members)] = sorted(members)
        for f in members:
            firm_unit[f] = u
            occ[u, f // N] = True
    home = np.repeat(np.arange(M, dtype=np.int16), N)
    NB = cg.K_W_BINS + 1
    z = lambda *s, t=np.float64: np.zeros(s, t)  # noqa: E731
    cg.seed_numba(seed)
    history = []
    for step in range(window, window + steps):
        out = cg.merger_kernel_numba(
            step, window, 0.3, 0.5, firm_unit, home, np.full(F, -1, np.int32), unit_members, unit_size, unit_active,
            occ, np.zeros((MAXU, window)), np.array([0.0] + [1e-5] * M), buf, ret, F, M, N, 1, ACCEPT, 0.055,
            np.zeros(MAXU, np.int32), np.full(MAXU, 0.3), np.zeros(MAXU, np.int32),
            z(M + 1, cg.BINDING_GAIN_BINS, t=np.int64), True,
            z(M + 1, NB, t=np.int64), z(M + 1, NB, t=np.int64), z(M + 1, NB), z(M + 1, NB), z(M + 1, NB, t=np.int64),
            z(M + 1, NB), float(np.log(0.2 / N)),
            z(M + 1, NB, 3, t=np.int64), z(M + 1, NB, 3, t=np.int64), z(M + 1, NB, 3), z(M + 1, NB, 3),
            z(M + 1, NB, 3, t=np.int64),
            z(M + 1, 2, t=np.int64), z(M + 1, 2, t=np.int64), z(M + 1, 2), z(M + 1, 2), z(M + 1, 2, t=np.int64),
            z(M + 1, 2, t=np.int64), np.full(M, 0.2))
        n_merg, n_prop = out[0], out[1]
        assert n_merg == n_prop
        # occupancy invariant: every unit holds at most one firm per market, and occ matches its members
        for u in range(MAXU):
            k = unit_size[u]
            if k == 0:
                continue
            members = unit_members[u, :k]
            mk = home[members]
            assert len(set(mk)) == k, (step, u, members)
            assert np.array_equal(np.flatnonzero(occ[u]), np.sort(mk))
            assert (firm_unit[members] == u).all()
        history.append((n_prop, n_merg))
    return history, firm_unit, unit_members, unit_size, home


def test_accept_on_evolving_economy_never_merges_an_infeasible_pair():
    """(a) On an evolving five-market economy (two firms per market, mergers persist), every period's proposals
    are all merged, and no unit ever holds two firms of one market (checked after every period in
    _evolving_accept); units of three or more members form."""
    M, N = 5, 2
    sizes = np.log(np.linspace(0.5, 2.0, M * N))
    hist, firm_unit, unit_members, unit_size, home = _evolving_accept(sizes, [], M, N, steps=200, seed=3)
    assert sum(h[1] for h in hist) > 0
    assert unit_size.max() >= 3


def test_feasibility_oracle_on_the_three_market_economy():
    """The brute-force oracle agrees with the kernel on which pairs exist: under accept the fixed economy's
    proposals are exactly the feasible ones, so their K values are those of feasible pairs (2 for solo+solo, 3 for a
    standalone in market 2 joining unit {0, 2}); an infeasible pair (K = 3 with firm 1 or 3, or 4 with 5) never
    appears as a merger."""
    home = {f: f // 2 for f in range(6)}
    units_of = {0: (0, 2), 2: (0, 2)}
    pairs = [(a, b) for a in range(6) for b in range(6) if home[a] != home[b]]
    feas = {(a, b) for a, b in pairs if feasible(units_of, home, a, b)}
    assert (1, 2) not in feas and (3, 0) not in feas and (4, 0) in feas and (1, 3) in feas
    a4, props4, mergs4 = kernel(LOG_SIZES, units=UNITS, N=2, rule=ACCEPT, steps=200)
    assert set(np.flatnonzero(mergs4)) == {2, 3}
    # K = 3 mergers are solo+unit with a market-2 standalone (the only feasible K = 3 pairs)
    assert a4['mt'][3, :, 1].sum() == mergs4[3] and a4['mt'][3, :, [0, 2]].sum() == 0


def test_exit_kernel_returns_nothing_under_accept():
    """(b) compute_exit_candidates_numba: a member whose moment gain is negative leaves under the moment rule and
    not under accept."""
    F, window = 2, 10
    rng = np.random.default_rng(0)
    buf = np.log(np.tile([1.0, 1.0], (window + 1, 1)))
    ret = np.column_stack([rng.normal(0.055, 0.01, window + 1), rng.normal(0.055, 0.6, window + 1)])
    firm_unit = np.array([0, 0], np.int32)
    unit_members = np.array([[0, 1]], np.int32)
    unit_size = np.array([2], np.int16)
    phi = np.array([0.0, 0.0, 0.05])
    args = (np.zeros((window, F)), buf, ret, np.array([0, 1], np.int64), window, window, None, 0.055, 0.3, 1, phi,
            firm_unit, unit_members, unit_size, np.array([0.3]))
    moment = cg.compute_exit_candidates_numba(*args[:6], 2, *args[7:])
    accept = cg.compute_exit_candidates_numba(*args[:6], ACCEPT, *args[7:])
    assert len(moment) > 0 and len(accept) == 0


@pytest.fixture(scope='module')
def accept_run():
    return cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, decision_rule='accept', burn_in=0))[-1]


def test_no_voluntary_exit(accept_run):
    """(b) Over the whole run every exit from a unit is a floor exit; units form, and floor exits occur."""
    s = accept_run['summary']
    assert accept_run['mergers_by_K'].sum() > 0
    assert accept_run['floor_exits_per_period'].sum() > 0
    assert s['exits_per_period'] == s['floor_exits_per_period']


def test_accept_rule_validation_and_difference_from_moment(accept_run):
    """The accept run differs from the moment run on the same seed; accept is not combined with endogenous alpha;
    it does not need g (but the pilot passes it)."""
    hp_m = cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, decision_rule='moment', burn_in=0))[-1]
    assert not np.array_equal(accept_run['final_log_states'], hp_m['final_log_states'])
    assert accept_run['mergers_by_K'].sum() > hp_m['mergers_by_K'].sum()
    # every feasible proposal after step 0 is merged
    assert np.array_equal(accept_run['mergers_by_K'], accept_run['proposals_by_K'])
    with pytest.raises(ValueError):
        cg.model(**PARAMS, **dict(KW, decision_rule='accept', alpha_endogenous=True))
    with pytest.raises(ValueError):
        cg.model(**PARAMS, **dict(KW, decision_rule='unanimous'))


@pytest.mark.parametrize('sharing_rule', ['proportional', 'equal'])
def test_pooling_identity_under_accept(sharing_rule):
    """(d) Every active unit in every period of an accept run conserves sum_j s_j (1 + r_j) - Phi(K) S to 1e-9 S
    (limited liability: capped members' shortfalls are charged to the other members)."""
    hp, rec = _conservation_run(sharing_rule=sharing_rule, decision_rule='accept')
    assert rec['units'] > 0
    assert rec['max_rel_err'] < 1e-9, rec
    assert hp['summary']['pool_wipeout_total'] == 0


def test_identity_c42():
    """(c) Every existing pilot D scenario (2,055: C37, C38 alpha-0.9, C39 cv), rerun on the C42 code, is identical
    to its C39 run in every field and in its tidy row, with no new keys."""
    i = pd.read_csv(os.path.join(ROOT, 'diagnostics', 'identity_c42.csv'))
    assert len(i) == 2055 and sorted(i.scenario_id) == list(range(2055))
    assert i.identical.all() and i.identical_tidy.all()
