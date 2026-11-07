"""C38 Task 2: proposals, mergers and binding gain by K x entrant capital share w (merger_kernel_numba)."""
import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW

NB = cg.K_W_BINS + 1
C38 = ('proposals_by_K_w', 'mergers_by_K_w', 'binding_sum_by_K_w', 'binding_sq_by_K_w', 'binding_n_by_K_w',
       'w_sum_by_K_w')


def kernel(log_sizes, units=(), M=None, N=1, floor_c=0.1, rule=2, steps=40, seed=0):
    """Call merger_kernel_numba directly for `steps` periods on a fixed state (the state is restored after every
    call, so every call proposes on the same economy). log_sizes: log state of each firm (firm f in market f // N);
    units: tuples of firm ids forming units. Returns the C38 arrays and the kernel's proposals_by_K summed."""
    F = len(log_sizes)
    M = M or F // N
    window = 10
    rng = np.random.default_rng(seed)
    buf = np.tile(np.asarray(log_sizes, float), (window + 1, 1))
    ret = rng.normal(0.055, 0.05, (window + 1, F))
    MAXU = max(F // 2, 1)
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
    arrays = dict(p=np.zeros((M + 1, NB), np.int64), m=np.zeros((M + 1, NB), np.int64),
                  s1=np.zeros((M + 1, NB)), s2=np.zeros((M + 1, NB)), n=np.zeros((M + 1, NB), np.int64),
                  w=np.zeros((M + 1, NB)))
    props = np.zeros(M + 1, np.int64)
    mergs = np.zeros(M + 1, np.int64)
    cg.seed_numba(seed)
    for step in range(window, window + steps):
        state = (firm_unit.copy(), unit_members.copy(), unit_size.copy(), unit_active.copy(), occ.copy())
        out = cg.merger_kernel_numba(
            step, window, 0.3, 1.0, state[0], home, np.full(F, -1, np.int32), state[1], state[2], state[3],
            state[4], np.zeros((MAXU, window)), np.array([0.0] + [1e-5] * M), buf, ret, F, M, N, 1, rule, 0.055,
            np.zeros(MAXU, np.int32), np.full(MAXU, 0.3), np.zeros(MAXU, np.int32),
            np.zeros((M + 1, cg.BINDING_GAIN_BINS), np.int64), True,
            arrays['p'], arrays['m'], arrays['s1'], arrays['s2'], arrays['n'], arrays['w'],
            float(np.log(2 * floor_c / N)))
        props += out[8]
        mergs += out[9]
    return arrays, props, mergs


def test_solo_plus_unit_entrant_is_the_standalone():
    """Unit {0, 1} (sizes 1, 2) and standalone 2 (size 1.2), one firm per market: every proposal is solo+unit with
    K = 3 and w = 1.2 / 4.2, column 1 + 5. Catches: the incumbent side taken as entrant, or w on the merged unit's
    sizes in the wrong order."""
    a, props, mergs = kernel(np.log([1.0, 2.0, 1.2]), units=[(0, 1)])
    assert props[3] > 0 and props.sum() == props[3]
    assert a['p'][3, 6] == props[3] and a['p'].sum() == props.sum()
    assert np.isclose(a['w'][3, 6], 1.2 / 4.2 * props[3])
    assert np.array_equal(a['m'].sum(1), mergs)


def test_solo_plus_solo_entrant_is_the_smaller():
    """Two standalones of sizes 3 and 1.2: w = 1.2 / 4.2 whichever initiates."""
    a, props, _ = kernel(np.log([3.0, 1.2]))
    assert props[2] > 0 and a['p'][2, 6] == props[2] and a['p'].sum() == props[2]


def test_unit_plus_unit_entrant_is_the_smaller_unit():
    """Units {0, 1} (capital 2.4) and {2, 3} (capital 6): w = 2.4 / 8.4 for the K = 4 proposals."""
    a, props, _ = kernel(np.log([1.0, 1.4, 3.0, 3.0]), units=[(0, 1), (2, 3)])
    assert props[4] > 0 and a['p'][4, 6] == props[4] and a['p'].sum() == props[4]


def test_floor_column():
    """Two markets of two firms: firms 0 and 2 hold 1% of their market (below 2 c / N = 0.1), firms 1 and 3 the
    rest (sizes 99 and 261). A proposal with a small firm has it as entrant at the floor (column 0); 1 + 3 has
    entrant 1 with w = 99 / 360 = 0.275 (column 6). Catches: the floor test on the unit share w instead of the
    entrant's market share."""
    a, props, _ = kernel(np.log([1.0, 99.0, 3.0, 261.0]), N=2, floor_c=0.1, steps=200)
    used = np.nonzero(a['p'][2])[0]
    assert set(used) == {0, 6} and a['p'][2].sum() == props[2]
    assert np.isclose(a['w'][2, 6], 99 / 360 * a['p'][2, 6])


def test_binding_sums_under_moment():
    """Under the moment rule every proposal has a binding gain (finite here): n = proposals, sum^2 <= n sum sq."""
    a, props, _ = kernel(np.log([1.0, 2.0, 1.2]), units=[(0, 1)], rule=2)
    assert np.array_equal(a['n'].sum(1), props)
    n = a['n'][3, 6]
    assert a['s2'][3, 6] * n >= a['s1'][3, 6] ** 2 - 1e-18


@pytest.fixture(scope='module', params=['moment', 'loggain'])
def run(request):
    return cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, decision_rule=request.param))[-1]


def test_model_sums_over_w_equal_by_K(run):
    """Post-burn-in: summed over the w columns, the C38 counts equal proposals_by_K / mergers_by_K (C35); the
    binding count is at most the proposals, and the -inf ones (proposals - n) sit in the binding histogram's
    first bin; the mean w of each column lies in its bin; the floor column is used."""
    P, Mg, n = run['proposals_by_K_w'], run['mergers_by_K_w'], run['binding_n_by_K_w']
    assert P.shape == (PARAMS['M'] + 1, NB) and all(run[k].shape == P.shape for k in C38)
    assert np.array_equal(P.sum(1), run['proposals_by_K']) and np.array_equal(Mg.sum(1), run['mergers_by_K'])
    assert (Mg <= P).all() and (n <= P).all()
    assert np.all(P.sum(1) - n.sum(1) <= run['binding_gain_hist_by_K'][:, 0])
    with np.errstate(invalid='ignore'):
        wbar = run['w_sum_by_K_w'] / P
    e = run['K_w_edges']
    for j in range(1, NB):
        sel = P[:, j] > 0
        assert np.all((wbar[sel, j] >= e[j - 1] - 1e-12) & (wbar[sel, j] <= e[j] + 1e-12))
    assert P[:, 0].sum() > 0 and P[:, 1:].sum() > 0
    assert not P[:2].any()
