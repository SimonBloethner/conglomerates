"""C39 outputs: proposal type and volatility bin (Task 1), the cv block (Task 2), the identity check."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'pilot_d')


def read(name):
    return pd.read_csv(os.path.join(D, name))


def test_type_rows_sum_to_c38_rows():
    """Summed over type, the C39 rows of each run, K and w column equal the C38 rows of the same scenario
    (pilot_d/acceptance_by_K_w.csv, from the C38 pickles): the reruns are the same runs."""
    t, c = read('acceptance_by_K_w_type.csv'), read('acceptance_by_K_w.csv')
    assert t.scenario_id.nunique() == 15 and set(t.type) == {'solo+solo', 'solo+unit', 'unit+unit'}
    s = t.groupby(['scenario_id', 'K', 'w_bin'])[['proposals', 'mergers', 'binding_n']].sum().sort_index()
    c = c.set_index(['scenario_id', 'K', 'w_bin'])[['proposals', 'mergers', 'binding_n']].sort_index()
    assert s.index.equals(c.index) and (s == c).all().all()
    assert (t[t.type == 'solo+solo'].K == 2).all() and (t[t.type == 'unit+unit'].K >= 4).all()


def test_volbin_rows():
    v, t = read('unit_unit_by_volbin.csv'), read('acceptance_by_K_w_type.csv')
    assert set(v.volbin) == {'entrant calmer', 'entrant more volatile'}
    a = v.groupby(['scenario_id', 'K'])[['proposals', 'mergers']].sum().sort_index()
    b = t[t.type == 'unit+unit'].groupby(['scenario_id', 'K'])[['proposals', 'mergers']].sum().sort_index()
    assert a.index.equals(b.index) and (a == b).all().all()
    assert ((v.share_calm_binds >= 0) & (v.share_calm_binds <= 1)).all()


def test_cv_block_tables():
    c = read('cv_comparison.csv')
    assert len(c) == 9 and set(c.alpha) == {0.1, 0.3, 0.5}
    a = read('cv_acceptance_by_K.csv')
    assert set(a.rule) == {'cv', 'moment', 'loggain'} and (a.groupby(['family', 'rule']).size() == 11).all()
    t = read('tidy.csv')
    cv = t[t.block == 'cv']
    assert len(t) >= 2055 and len(cv) == 45 and list(cv.scenario_id) == list(range(2010, 2055))
    assert (cv.decision_rule == 'cv').all()


def test_approximation_error_cv():
    """With the weights frozen the cv estimate equals the replay's (the subtracted term sums to zero)."""
    e = read('approximation_error_cv.csv')
    assert len(e) == 48 and (e.max_abs_cv_fixed_minus_hat_fixed < 1e-12).all()
    assert (e[e.alpha == 'all'].n + e[e.alpha == 'all'].n_dropped == 18000).all()
    assert (e.rel_cv_moment_p5 <= e.rel_cv_moment_median).all() and (e.rel_cv_moment_median <= e.rel_cv_moment_p95).all()


def test_identity_c39():
    """Every pilot D scenario of C37 and C38 (2,010), rerun on the C39 code, is identical except the new keys."""
    i = pd.read_csv(os.path.join(ROOT, 'diagnostics', 'identity_c39.csv'))
    assert len(i) == 2010 and sorted(i.scenario_id) == list(range(2010))
    assert i.identical.all() and i.identical_tidy.all() and i.counters_consistent.all()
