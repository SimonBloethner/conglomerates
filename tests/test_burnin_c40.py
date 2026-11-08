"""C40: the C33 burn-in check under the moment rule (pilot D), T = 16000."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_selection_is_the_moment_twin_of_c33():
    """20 pilot D main-block power-law moment scenarios, the twins of C33's 20 pilot C runs (same seeds)."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        import run_burnin_c33
        import run_burnin_c40
        d, c = run_burnin_c40.burnin_scenarios(), run_burnin_c33.burnin_scenarios()
    finally:
        os.chdir(cwd)
    assert [s['scenario_id'] for s in d] == [*range(315, 320), *range(335, 340), *range(345, 350), *range(525, 530)]
    assert all(s['decision_rule'] == 'moment' and s['block'] == 'main' and s['cost_type'] == 'power_law' for s in d)
    assert [s['twin_id'] for s in d] == [s['scenario_id'] for s in c]
    assert [s['seed'] for s in d] == [s['seed'] for s in c]
    assert {s['T'] for s in d} == {11000} and {s['burn_in'] for s in d} == {8000}


def read(name):
    return pd.read_csv(os.path.join(ROOT, 'diagnostics', name), float_precision='round_trip')


def test_series_csv():
    d = read('burn_in_c40.csv')
    assert len(d) == 20 * 160 and d.groupby(['log_family', 'alpha', 'rep']).size().eq(160).all()
    assert d.step.min() == 100 and d.step.max() == 16000


def test_alpha0_equals_c33():
    """At alpha = 0 no units form, so the moment-rule series equal the loggain ones run for run."""
    d, c = read('burn_in_c40.csv'), read('burn_in_c33_ext.csv')
    d0 = d[d.alpha == 0].sort_values(['rep', 'step']).reset_index(drop=True)
    c0 = c[c.alpha == 0].sort_values(['rep', 'step']).reset_index(drop=True)
    cols = [k for k in c.columns if k not in ('log_family', 'alpha', 'rep', 'seed', 'step')]
    assert (d0.seed == c0.seed).all() and (d0.step == c0.step).all()
    for k in cols:
        assert np.array_equal(d0[k].to_numpy(), c0[k].to_numpy(), equal_nan=True), k


def test_report_and_a3():
    md = open(os.path.join(ROOT, 'diagnostics', 'burn_in_c40.md')).read()
    assert '20 of 20 runs reproduce their stored pilot D result' in md
    a3 = open(os.path.join(ROOT, 'latex', 'tables', 'appendix', 'a3_burnin.tex')).read()
    d = read('burn_in_c40.csv')
    g = d[(d.log_family == 't3') & (d.alpha == 0.3)]
    pilot = np.mean([r[(r.step > 8000) & (r.step <= 11000)].ccs.mean() for _, r in g.groupby('rep')])
    assert f'$t_3$ & 0.3 & Pooled capital share & {pilot:.4f}' in a3
