"""C37: pilot D outputs (pilot_d/*.csv). Pilot C's outputs are checked unchanged by tests/test_tidy_schema.py and
tests/test_pilot_c_results.py, which still read pilot_c/."""
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'pilot_d')

from tests.test_pilot_d_design import BLOCK_COUNTS, NEW_BLOCKS, TOTAL, TOTAL_C37  # noqa: E402
from tests.test_tidy_schema import EXPECTED_TIDY_COLUMNS  # noqa: E402

EXTRA = ['pilot', 'twin_id', 'source_id', 'nu', 'iqr_lo', 'iqr_hi', 'member_firm_share',
         'floor_hit_rate_member_status', 'floor_hit_rate_standalone_status']


@pytest.fixture(scope='module')
def tidy():
    return pd.read_csv(os.path.join(D, 'tidy.csv'))


def test_tidy_rows_blocks_columns(tidy):
    assert len(tidy) == TOTAL
    assert tidy.block.value_counts().to_dict() == BLOCK_COUNTS
    assert sorted(tidy.scenario_id) == list(range(TOTAL))
    assert list(tidy.columns) == EXPECTED_TIDY_COLUMNS_ORDERED(tidy) and set(EXTRA) <= set(tidy.columns)
    assert (tidy.decision_rule == 'moment').sum() == TOTAL - 60


def EXPECTED_TIDY_COLUMNS_ORDERED(tidy):
    """Pilot C's columns (in pilot C's order), then the pilot D extras."""
    c = list(pd.read_csv(os.path.join(ROOT, 'pilot_c', 'tidy.csv'), nrows=0).columns)
    assert set(EXPECTED_TIDY_COLUMNS) <= set(c)
    return c + EXTRA


def test_status_rates_consistent(tidy):
    """All-firm rate = per-status rate x status share, for both statuses (where defined)."""
    t = tidy[(tidy.member_firm_share > 0) & (tidy.member_firm_share < 1)]
    assert np.allclose(t.floor_hit_rate_member_status * t.member_firm_share, t.floor_hit_rate_member)
    assert np.allclose(t.floor_hit_rate_standalone_status * (1 - t.member_firm_share), t.floor_hit_rate_standalone)
    assert (tidy[tidy.alpha == 0].member_firm_share == 0).all()


def test_homogeneous_and_nu_rows(tidy):
    h = tidy[tidy.block == 'homogeneous']
    assert (h.iqr_lo == 0.2).all() and (h.iqr_hi == 0.2).all()
    assert h.decision_rule.value_counts().to_dict() == {'moment': 60, 'loggain': 60}
    n = tidy[tidy.block == 'nu']
    assert sorted(n.nu.unique()) == [2.5, 4.0] and (n.log_family == 'student_t').all()
    assert np.isfinite(n[['K_mean', 'hill_exponent_median', 'floor_hit_rate_member']]).all().all()


def test_medians_cover_every_cell(tidy):
    m = pd.read_csv(os.path.join(D, 'medians.csv'))
    assert m.n_reps.sum() == TOTAL_C37          # C37 output, not regenerated in C38
    # every cell has 5 reps: pilot D groups by nu and floor_c too (pilot C's medians pool the two floor levels)
    assert (m.n_reps == 5).all()
    assert len(m) == TOTAL_C37 // 5


def test_provenance_and_copies():
    p = pd.read_csv(os.path.join(D, 'provenance.csv'))
    assert len(p) == TOTAL_C37
    assert (p.origin.str.startswith('copied')).sum() == 255 and p[p.source_id.notna()].source_md5.str.len().eq(32).all()
    chk = pd.read_csv(os.path.join(ROOT, 'diagnostics', 'identity_c37_copies.csv'))
    assert len(chk) == 14 and chk.identical.all() and chk.same_scenario.all()


def test_comparison_pairs_every_twin():
    c = pd.read_csv(os.path.join(D, 'comparison_c_vs_d.csv'))
    assert c.n_pairs.sum() == 1845 and set(c.block) == set(BLOCK_COUNTS) - set(NEW_BLOCKS)
    for col in ('K_mean_diff_median', 'K_mean_diff_p25', 'K_mean_diff_p75', 'acceptance_rate_diff_median',
                'floor_hit_rate_member_status_loggain_median'):
        assert col in c.columns
    k = c[c.K_mean_diff_median.notna()]          # K_mean is NaN in runs without any unit (alpha = 0 and a few more)
    assert len(k) > 300
    assert ((k.K_mean_diff_p25 <= k.K_mean_diff_median) & (k.K_mean_diff_median <= k.K_mean_diff_p75)).all()


def test_trace_outputs():
    for name, n in (('status_panel_summary.csv', 27), ('status_panel_decomposition.csv', 27),
                    ('event_study_summary.csv', 27), ('burstiness_summary.csv', 27)):
        df = pd.read_csv(os.path.join(D, name))
        assert len(df) == n, name
    dec = pd.read_csv(os.path.join(D, 'status_panel_decomposition.csv'))
    assert dec.check_floor_jump_aligned.all() and dec.check_floor_matches_pickle.all()
    assert dec.decomp_residual.abs().max() < 1e-12


def test_acceptance_and_alpha_scatter():
    a = pd.read_csv(os.path.join(D, 'acceptance_by_K_pooled.csv'))
    assert set(a.block) == {'main', 'search', 'cost-level-high', 'homogeneous'}
    assert set(a[a.block == 'homogeneous'].decision_rule) == {'moment', 'loggain'}
    s = pd.read_csv(os.path.join(D, 'alpha_scatter.csv'))
    assert s.scenario_id.nunique() <= 60 and len(s) > 0


def test_variance_ratio():
    v = pd.read_csv(os.path.join(D, 'variance_ratio.csv'))
    t = v[(v.family == 't') & v.source.str.startswith('iid')]
    assert sorted(t.nu) == [2.5, 3.0, 4.0]
    med = t.set_index('nu').q50
    assert med[2.5] < med[3.0] < med[4.0] < 1          # heavier tails: lower median ratio
    # the mean ratio is 1 in expectation; with nu = 2.5 the window variance has no finite variance itself, so its
    # sample mean converges too slowly to test
    iid = v[v.source.str.startswith('iid') & (v.nu.isna() | (v.nu > 2.5))]
    assert np.allclose(iid['mean'], 1, atol=0.05)
