"""C42 Task 1: rank mobility and persistence within markets (analysis/rank_mobility.py)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import rank_mobility as rm  # noqa: E402

T = 700


def make(ls, cong=None, floor=None):
    """Trace dict from a (T + 1, F) log-share array."""
    F = ls.shape[1]
    return dict(logshare=ls.astype(np.float32),
                cong=np.full(ls.shape, -1, np.int16) if cong is None else cong,
                floor=np.zeros((F, ls.shape[0] - 1), bool) if floor is None else floor)


def values(rows):
    return {(r['h'], r['sample'], r['status'], r['measure']): r['value'] for r in rows}


def test_frozen_ranks():
    """Shares that never change rank: Spearman 1, deciles persist, no leader turnover, no rank change."""
    M, N = 2, 20
    base = np.log(np.arange(1, M * N + 1, dtype=float))
    ls = np.tile(base, (T + 1, 1)) + 0.01 * np.arange(T + 1)[:, None]
    v = values(rm.trace_rows(make(ls), M, N, 0, {}))
    for h in rm.LAGS:
        assert v[(h, 'all', 'all', 'spearman')] == pytest.approx(1.0)
        assert v[(h, 'all', 'all', 'top_persist')] == 1.0 and v[(h, 'all', 'all', 'bottom_persist')] == 1.0
        assert v[(h, 'all', 'all', 'leader_turnover')] == 0.0 and v[(h, 'all', 'all', 'mean_abs_drank')] == 0.0
        assert v[(h, 'owners', 'all', 'spearman')] == pytest.approx(1.0)
        assert v[(h, 'owners', 'all', 'top_persist')] == 1.0 and v[(h, 'owners', 'all', 'leader_turnover')] == 0.0


def test_reversal():
    """Ranks reversed in alternate blocks of 50 periods (origins are multiples of 100, so t + 50 is reversed and
    t + 100 is not): at h = 50 Spearman -1, no decile persistence, the leader always replaced, mean |rank change|
    n / 2 (0..n-1 reversed, n even); at h = 100 Spearman 1."""
    M, N = 1, 20
    base = np.log(np.arange(1, N + 1, dtype=float))
    h = 50
    ls = np.where(((np.arange(T + 1) // 50) % 2 == 0)[:, None], base, base[::-1])
    v = values(rm.trace_rows(make(ls), M, N, 0, {}))
    assert v[(h, 'all', 'all', 'spearman')] == pytest.approx(-1.0)
    assert v[(h, 'all', 'all', 'top_persist')] == 0.0 and v[(h, 'all', 'all', 'leader_turnover')] == 1.0
    assert v[(h, 'all', 'all', 'mean_abs_drank')] == pytest.approx(N / 2)
    assert v[(100, 'all', 'all', 'spearman')] == pytest.approx(1.0)


def test_floor_lift_moves_only_the_all_sample():
    """One market of 20; firm 0 (the smallest) is lifted at step t + 3 to the top for good. In the all sample its rise
    pushes every other firm down one rank and replaces the leader; in the hit-free sample firm 0 is left out and
    the others' ranks do not move."""
    M, N = 1, 20
    base = np.log(np.arange(1, N + 1, dtype=float))
    ls = np.tile(base, (T + 1, 1))
    floor = np.zeros((N, T), bool)
    t_jump = 103
    ls[t_jump + 1:, 0] = np.log(100.0)
    floor[0, t_jump] = True
    v = values(rm.trace_rows(make(ls, floor=floor), M, N, 0, {}))
    h = 10  # origin t = 100: lift in step 103 is in (100, 110]
    n_orig = len(rm.origins(T, 0, h))
    assert v[(h, 'all', 'all', 'leader_turnover')] == pytest.approx(1 / n_orig)
    assert v[(h, 'all', 'all', 'hit_share')] == pytest.approx(1 / (N * n_orig))
    assert v[(h, 'hitfree', 'all', 'spearman')] == pytest.approx(1.0)
    assert v[(h, 'hitfree', 'all', 'leader_turnover')] == 0.0 and v[(h, 'hitfree', 'all', 'mean_abs_drank')] == 0.0
    assert v[(h, 'all', 'all', 'spearman')] < 1.0


def test_status_split_and_owners():
    """Two markets of 10. Firms 9 (market 0) and 19 (market 1), the leaders, form a unit at all times: members are the
    top of each market (members' share of the top decile 1) and persist; the unit is the economy's largest owner,
    with capital share 2 * s_9 / M."""
    M, N = 2, 10
    base = np.log(np.r_[np.arange(1, N + 1), np.arange(1, N + 1)].astype(float))
    ls = np.tile(base, (T + 1, 1))
    cong = np.full(ls.shape, -1, np.int16)
    cong[:, [9, 19]] = 0
    v = values(rm.trace_rows(make(ls, cong=cong), M, N, 0, {}))
    h = 100
    assert v[(h, 'all', 'member', 'top_share_of_decile')] == 1.0
    assert v[(h, 'all', 'member', 'top_persist')] == 1.0
    ids, inv, cap, anchor = rm.owners_at(ls[0], cong[0])
    assert len(cap) == 19 and cap.max() == pytest.approx(2 * 10.0)
    assert v[(h, 'owners', 'all', 'n_owners')] == 19
    # top decile of 19 owners is the single largest: the unit, with 20 of the 110 capital
    assert v[(h, 'owners', 'all', 'top_share')] == pytest.approx(20 / 110)


def test_owner_capital_and_anchor():
    """Owners: unit 0 (firms 2, 3; capital 8, anchor firm 2) and standalones 0 and 1 (owner ids F + firm)."""
    ls = np.log(np.array([[1.0, 2.0, 5.0, 3.0]]))
    cong = np.array([[-1, -1, 0, 0]], np.int16)
    ids0, inv0, cap0, anchor0 = rm.owners_at(ls[0], cong[0])
    assert sorted(cap0) == pytest.approx([1.0, 2.0, 8.0])
    assert ls[0][anchor0].tolist() == pytest.approx(np.log([5.0, 1.0, 2.0]).tolist())  # unit 0 first, then 4 + 0, 4 + 1


def test_output():
    p = os.path.join(ROOT, 'pilot_d', 'rank_mobility.csv')
    df = pd.read_csv(p)
    assert set(df.family) == {'normal', 'laplace', 't3'} and set(df.h) == set(rm.LAGS)
    assert {0.05, 0.1, 0.3} <= set(df.alpha)
    assert df.groupby(['family', 'alpha']).rep.nunique().eq(3).all()
    assert (df[df.value.isna()].n == 0).all()          # undefined only where there is nothing to average
    df = df.dropna(subset=['value'])
    s = df[df.measure == 'spearman']
    assert s.value.between(-1, 1).all()
    p = df[df.measure.isin(['top_persist', 'bottom_persist', 'leader_turnover', 'hit_share'])]
    assert p.value.between(0, 1).all()
    # alpha = 0: no members; their share of the top decile (denominator: the whole decile) is 0
    a0 = df[(df.alpha == 0) & (df.status == 'member')]
    assert len(a0) and (a0[a0.measure == 'top_share_of_decile'].value == 0).all()
    assert len(df[(df.alpha == 0) & (df.status == 'member') & (df.measure != 'top_share_of_decile')]) == 0


def test_range_sd_matches_the_models_online_statistics():
    """The trace-based per-firm rank range and rank SD over rows 1..T equal the model's online rank_range and
    rank_std (the May 2025 article's measures) on a traced run without a floor (no ties at the floor)."""
    import tempfile
    import collaborative_growth as cg
    from tests.test_floor_exit import PARAMS, KW
    with tempfile.TemporaryDirectory() as d:
        tp = os.path.join(d, 't.npz')
        res = cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, floor_c=0.0, trace_path=tp))
        ls = np.load(tp)['logshare']
    M, N = PARAMS['M'], PARAMS['N']
    rr, sd = rm.firm_range_sd(ls, M, N, 1, ls.shape[0])
    model_rr, model_sd = np.asarray(res[10], float).ravel(), np.asarray(res[11], float).ravel()
    assert np.mean(rr == model_rr) > 0.99
    assert np.mean(np.abs(sd - model_sd) < 1e-3) > 0.99


def test_range_sd_by_hand():
    """Two markets of 3 firms over 4 rows: firm 0 ranks 1, 3, 2 (rows 1..3) -> range 2, sd sqrt(2/3)."""
    ls = np.log(np.array([[1, 2, 3, 1, 2, 3],
                          [1, 2, 3, 1, 2, 3],
                          [5, 2, 3, 1, 2, 3],
                          [2.5, 2, 3, 1, 2, 3]], float))
    rr, sd = rm.firm_range_sd(ls, 2, 3, 1, 4, chunk=2)
    assert rr[0] == 2 and sd[0] == pytest.approx(np.sqrt(2 / 3))
    assert rr[3] == 0 and sd[3] == 0
