"""C38 outputs: event-study matchings (Task 1), acceptance by K x entrant share (Task 2), approximation error
(Task 3), the fixed alpha = 0.9 cell (Task 4) and the identity check of the C38 counters."""
import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'pilot_d')


def read(name):
    return pd.read_csv(os.path.join(D, name))


def test_event_study_matching():
    """Six matchings x 9 cells (3 reps each); the means decompose exactly (did = post gap - pre gap) per trace;
    the hit-free window keeps fewer events the longer it is; the current matching keeps every event."""
    m, r = read('event_study_matching.csv'), read('event_study_matching_by_rep.csv')
    assert len(m) == 54 and (m.n_reps == 3).all() and len(r) == 162
    assert set(m.variant) == {'current', 'A_owner', 'B_k5', 'B_k10', 'B_k20', 'C_recap'}
    assert np.allclose(r.did_owner_mean, r.post_gap_owner_mean - r.pre_gap_owner_mean, atol=1e-12)
    w = r.pivot_table(index=['family', 'alpha', 'rep'], columns='variant', values='n_events')
    assert (w.current >= w.B_k5).all() and (w.B_k5 >= w.B_k10).all() and (w.B_k10 >= w.B_k20).all()
    assert (w.current == w.A_owner).all() and (w.current == w.C_recap).all()
    s = read('event_study_summary.csv').set_index(['family', 'alpha', 'rep'])
    cur = r[r.variant == 'current'].set_index(['family', 'alpha', 'rep'])
    assert np.allclose(cur.did_owner_median, s.loc[cur.index].did_owner_median)


def test_acceptance_by_K_w_sums_to_by_K():
    """Summed over the w columns, the C38 proposals and mergers of each run and K equal the C35/C37 counts of the
    same scenario in pilot_d/acceptance_by_K.csv (an independent output of the C37 pickles)."""
    a = read('acceptance_by_K_w.csv')
    assert a.scenario_id.nunique() == 15 and (a.alpha == 0.3).all()
    assert set(a.family) == {'normal', 'laplace', 't3'}
    s = a.groupby(['scenario_id', 'K'])[['proposals', 'mergers']].sum()
    c = read('acceptance_by_K.csv').set_index(['scenario_id', 'K'])
    c = c[c.proposals > 0]
    common = s.index.intersection(c.index)
    assert len(common) == len(c.loc[c.index.get_level_values(0).isin(a.scenario_id.unique())])
    assert (s.loc[common].proposals == c.loc[common].proposals).all()
    assert (s.loc[common].mergers == c.loc[common].mergers).all()
    floor = a[a.w_bin == 'floor']
    assert len(floor) and floor.w_lo.isna().all()
    assert ((a.binding_n <= a.proposals) & (a.mergers <= a.proposals)).all()
    p = read('acceptance_by_K_w_pooled.csv')
    assert np.allclose(p.incumbent_margin, p.gain_w_incumbent - p.marginal_cost)


def test_approximation_error():
    """3 families x K in {2, 3, 5, 8}, pooled (18,000 draws) and by alpha (6,000); the mean difference splits into
    the three terms exactly."""
    e = read('approximation_error.csv')
    assert len(e) == 48 and set(e.K) == {2, 3, 5, 8}
    assert (e[e.alpha == 'all'].n + e[e.alpha == 'all'].n_dropped == 18000).all()
    assert (e[e.alpha != 'all'].n + e[e.alpha != 'all'].n_dropped == 6000).all()
    tot = e.higher_order_mean + e.weight_term_hat_mean + e.weight_term_tilde_mean
    assert np.allclose(e.diff_mean, tot, rtol=1e-9, atol=1e-15)
    assert (e.diff_p5 <= e.diff_median).all() and (e.diff_median <= e.diff_p95).all()


def test_fixed_alpha_table():
    f = read('fixed_alpha_09.csv')
    assert len(f) == 9 and (f.n_reps == 5).all()
    t = read('tidy.csv')
    a9 = t[t.block == 'alpha-0.9']
    assert len(a9) == 15 and (a9.alpha == 0.9).all() and list(a9.scenario_id) == list(range(1995, 2010))


def test_identity_c38():
    """Every pilot D scenario rerun on the C38 code is identical to its C37 result without the C38 keys."""
    i = pd.read_csv(os.path.join(ROOT, 'diagnostics', 'identity_c38.csv'))
    assert len(i) == 1995 and sorted(i.scenario_id) == list(range(1995))
    assert i.identical.all() and i.identical_tidy.all() and i.counters_consistent.all()
