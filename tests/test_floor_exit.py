"""C27: a firm that hits the floor leaves its conglomerate in the same step."""
import numpy as np
import pytest

import collaborative_growth as cg

M = N = 20
T = 600
PARAMS = [M, N, T, 0.3, M * N, 0.05, 4, 0.0, False, 50, 'power_law', 2.032e-05, 1.2, 0.001]
KW = dict(seed=742, growth_process='log_family', log_family='laplace', mu_range=(0.01, 0.1),
          sigma_range=(0.1, 0.3), floor_c=0.3, cross_corr=0.0, metric_every=100,
          sharing_rule='proportional', burn_in=100, g=0.055, market_size_fixed=True,
          decision_rule='loggain')


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """One seeded run; wraps exit_ to check occupancy after every exit (incl. floor exits)."""
    exit_checks = {'calls': 0, 'violations': []}
    orig_exit = cg.exit_

    def checked_exit(firm_id, firm_conglom, firm_entered, firm_home_market, cong_firms, cong_size,
                     cong_occupies_market, **kw):
        cid = int(firm_conglom[firm_id])
        orig_exit(firm_id, firm_conglom, firm_entered, firm_home_market, cong_firms, cong_size,
                  cong_occupies_market, **kw)
        exit_checks['calls'] += 1
        if cid != -1 and cong_size[cid] >= 2:
            members = cong_firms[cid, :cong_size[cid]]
            homes = firm_home_market[members]
            occupied = set(np.nonzero(cong_occupies_market[cid])[0].tolist())
            if len(set(homes.tolist())) != len(homes) or occupied != set(homes.tolist()):
                exit_checks['violations'].append((kw.get('step'), cid))

    trace_path = str(tmp_path_factory.mktemp('c27') / 'trace.npz')
    cg.exit_ = checked_exit
    try:
        res = cg.model(PARAMS, trace_path=trace_path, **KW)
    finally:
        cg.exit_ = orig_exit
    return res[-1], np.load(trace_path), exit_checks


def test_floor_hit_leaves_conglomerate(run):
    """(a) floor[i, t] True implies cong[t+1, i] == -1 (trace row t+1 is the state after step t)."""
    _, z, _ = run
    floor, cong = z['floor'], z['cong']
    assert floor.shape == (M * N, T)
    firms, steps = np.nonzero(floor)
    assert len(firms) > 0, 'no floor hits; test is vacuous'
    assert (cong[steps + 1, firms] == -1).all()


def test_floor_exit_counts(run):
    """(b) 0 < sum(floor_exits_per_period) <= sum(floor_hits)."""
    hp, _, _ = run
    fe = hp['floor_exits_per_period'].sum()
    assert 0 < fe <= hp['floor_hits'].sum()


def test_occupancy_invariant(run):
    """(c) Distinct member home markets: checked at every step from the trace (cong, home).
    cong_occupies_market == set of member home markets: the model does not return it per step,
    so it is checked after every exit_ call (which includes all C27 floor exits) via a wrapper;
    merger-kernel updates are covered only indirectly by the per-step distinct-home check.
    At T the final membership (final_firm_conglom) is checked as well."""
    hp, z, exit_checks = run
    cong, home = z['cong'], z['home']
    for t in range(cong.shape[0]):
        row = cong[t]
        ids = row[row >= 0]
        for cid in np.unique(ids):
            h = home[row == cid]
            assert len(h) >= 2, (t, cid)
            assert len(np.unique(h)) == len(h), (t, cid)
    final = hp['final_firm_conglom']
    assert (final == cong[-1]).all()
    assert exit_checks['calls'] > 0
    assert exit_checks['violations'] == []
