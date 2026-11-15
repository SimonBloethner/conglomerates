"""C43: the alpha-gap block (main power-law cells at alpha 0.6, 0.7, 0.8; pilot D ids 2100-2144) in pilot_d/tidy.csv
and pilot_d/alpha_gap.csv, and the identity checks (diagnostics/identity_c43_*.csv)."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'pilot_d')
KEYS = ['cell_id', 'seed', 'cost_type', 'lookback', 'cross_corr', 'sharing_rule', 'merge_thresh', 'M', 'N',
        'floor_c', 'cost_multiplier', 'iqr_lo', 'iqr_hi', 'decision_rule']


def test_tidy_alpha_gap_rows():
    t = pd.read_csv(os.path.join(D, 'tidy.csv'))
    g = t[t.block == 'alpha-gap']
    assert len(g) == 45 and list(g.scenario_id) == list(range(2100, 2145))
    assert sorted(g.alpha.unique()) == [0.6, 0.7, 0.8] and (g.decision_rule == 'moment').all()
    assert g.twin_id.isna().all() and g.source_id.isna().all()
    main = t[(t.block == 'main') & (t.cost_type == 'power_law') & (t.alpha == 0.5)].set_index(['log_family', 'rep'])
    for _, r in g.iterrows():
        m = main.loc[(r.log_family, r.rep)]
        assert all(r[k] == m[k] or (pd.isna(r[k]) and pd.isna(m[k])) for k in KEYS)


def test_alpha_gap_csv():
    a = pd.read_csv(os.path.join(D, 'alpha_gap.csv'))
    assert len(a) == 45 and set(a.family) == {'normal', 'laplace', 't3'}
    for c in ('K_mean', 'cong_capital_share_median', 'hill_exponent_median', 'hhi_within_median',
              'floor_exits_per_period', 'voluntary_exits_per_period', 'acceptance_rate',
              'floor_hit_rate_member_status', 'floor_hit_rate_standalone_status', 'top10pct_aggregate_median',
              'K_eff_over_K_median', 'elapsed_seconds'):
        assert c in a.columns and a[c].notna().all(), c


def test_identity_checks():
    d = pd.read_csv(os.path.join(ROOT, 'diagnostics', 'identity_c43_design.csv'))
    assert len(d) == 2100 and d.identical.all()
    n = pd.read_csv(os.path.join(ROOT, 'diagnostics', 'identity_c43_alpha09.csv'))
    assert len(n) == 15 and n.identical.all()
