"""C30: owner-return growth by affiliation status (analysis/status_panel.py)."""
import numpy as np
import pytest

import collaborative_growth as cg
from analysis.status_panel import panel_from_trace, summarize
from tests.test_floor_exit import PARAMS, KW, M, N, T


def _hand_trace():
    """2 markets x 3 firms, T = 5 periods (6 trace rows).

    Firms 0 (market 0) and 3 (market 1) form one two-firm conglomerate that is in place at the start of
    periods 0-4 and dissolved in the last row; members grow by exactly 0.1 per period, standalones by 0.
    Standalone firm 2 is recapitalised by a jump of 0.5 in period 2 (floor hit), so its owner return is 0.
    Sizes interleave so that each market has a size-decile cell holding both statuses."""
    T5, F = 5, 6
    home = np.array([0, 0, 0, 1, 1, 1])
    t = np.arange(T5 + 1)[:, None]
    logshare = np.zeros((T5 + 1, F))
    logshare[:, [0]] = -1.2 + 0.1 * t
    logshare[:, [3]] = -1.2 + 0.1 * t
    logshare[:, 1] = -1.15
    logshare[:, 4] = -1.15
    logshare[:, 2] = -1.25
    logshare[:, 5] = -1.3
    jump = np.zeros((T5 + 1, F))
    floor = np.zeros((F, T5), dtype=bool)
    jump[3, 2] = 0.5                 # recorded at row t + 1 for period t = 2
    logshare[3:, 2] += 0.5
    floor[2, 2] = True
    cong = np.full((T5 + 1, F), -1, dtype=np.int16)
    cong[:T5, 0] = 0                 # member at the start of periods 0..4, standalone in the last row
    cong[:T5, 3] = 0
    return logshare, cong, floor, home, jump


def test_hand_built_trace():
    """(a) Exact values on a hand-built trace, to 1e-12.
    Catches: a jump subtracted on the wrong side (inc_standalone_mean would be 0.05), status taken at the
    end instead of the start of the period (period 4 of the members would count as standalone with
    inc 0.1, and 22 standalone firm-periods would give floor rate 1/22), and a K miscount (diff_K2_mean
    NaN, or diff_K5p_mean finite)."""
    df = panel_from_trace(*_hand_trace(), burn_in=0)
    out = summarize(df)
    assert out['n_member'] == 10 and out['n_standalone'] == 20
    assert abs(out['inc_standalone_mean'] - 0.0) < 1e-12
    assert abs(out['inc_member_mean'] - 0.1) < 1e-12
    assert abs(out['diff_fe_mean'] - 0.1) < 1e-12
    assert abs(out['diff_K2_mean'] - 0.1) < 1e-12
    assert abs(out['floor_rate_standalone'] - 0.05) < 1e-12
    assert np.isnan(out['diff_K5p_mean'])


@pytest.fixture(scope='module')
def seeded_panel(tmp_path_factory):
    """Panel on the seeded M=N=20, T=600 trace of tests/test_floor_exit.py, burn_in = 100."""
    assert KW['burn_in'] == 100
    trace_path = str(tmp_path_factory.mktemp('c30') / 'trace.npz')
    cg.model(PARAMS, trace_path=trace_path, **KW)
    z = np.load(trace_path)
    return panel_from_trace(z['logshare'], z['cong'], z['floor'], z['home'], z['jump'], burn_in=KW['burn_in'])


def test_seeded_panel_counts(seeded_panel):
    """(b) Row counts and decile weights on a model trace.
    Catches: dropping or double-counting firm-periods (n_rows, n_member + n_standalone), and decile weights
    that do not equal the member firm-periods in cells with both statuses."""
    df = seeded_panel
    out = summarize(df)
    n_rows = len(df)
    assert n_rows == 20 * 20 * 500 == M * N * (T - KW['burn_in'])
    assert out['n_member'] + out['n_standalone'] == n_rows

    n_dec = [out[f'n_dec{d}'] for d in range(1, 11)]
    assert all(n >= 0 for n in n_dec)
    has = df.groupby(['market', 'decile'])['member'].agg(['any', 'all'])
    both = has[has['any'] & ~has['all']].index
    cell = df.set_index(['market', 'decile'])
    n_member_both = int(cell.loc[cell.index.isin(both), 'member'].sum())
    assert sum(n_dec) == n_member_both


def test_seeded_floor_rates(seeded_panel):
    """(b) Members hit the floor less often than standalones.
    Catches: floor hits attributed to the wrong status (e.g. status at end of period, after a floor exit)."""
    out = summarize(seeded_panel)
    assert out['floor_rate_member'] < out['floor_rate_standalone']
