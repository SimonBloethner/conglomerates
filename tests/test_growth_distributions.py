"""C41: growth-rate distributions from the traces (analysis/growth_distributions.py)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from growth_distributions import growth_panel, stats  # noqa: E402
from collaborative_growth import process_conglomerate_pooling_numba  # noqa: E402


def log_shares(levels, M, N):
    """(T + 1, M * N) log levels -> within-market log shares, as the trace stores them."""
    x = levels.reshape(levels.shape[0], M, N)
    mx = x.max(axis=2, keepdims=True)
    return (x - (mx + np.log(np.exp(x - mx).sum(axis=2, keepdims=True)))).reshape(levels.shape)


def test_synthetic_laplace_alpha0_reproduces_the_shocks():
    """Standalones only (alpha = 0), i.i.d. Laplace shocks at IQR q: the measured growth has the shocks'
    statistics (sd = b sqrt 2, IQR = q, MAD/SD = 0.707, excess kurtosis 3, skew 0) within sampling error."""
    rng = np.random.default_rng(41)
    M, N, T, q, g = 20, 800, 30, 0.2, 0.05
    b = q / (2 * np.log(2))                                    # Laplace scale with IQR q
    shocks = g + rng.laplace(0.0, b, size=(T, M * N))
    levels = np.vstack([np.zeros(M * N), np.cumsum(shocks, axis=0)])
    trace = dict(logshare=log_shares(levels, M, N).astype(np.float32),
                 cong=np.full((T + 1, M * N), -1, dtype=np.int16),
                 floor=np.zeros((M * N, T), dtype=bool), home=np.repeat(np.arange(M), N))
    p = growth_panel(**trace, burn_in=0, q_m=np.full(M, q))
    assert p['n_excluded'] == 0 and not p['member'].any() and len(p['raw']) == T * M * N
    n = len(p['raw'])
    for meas, scale in (('raw', 1.0), ('demeaned', 1.0), ('standardized', 1 / q)):
        s = stats(p[meas])
        assert abs(s['sd'] / (b * np.sqrt(2) * scale) - 1) < 0.01, (meas, s['sd'])
        assert abs(s['iqr'] / (q * scale) - 1) < 0.02, (meas, s['iqr'])
        assert abs(s['mad_sd'] - np.sqrt(2) / 2) < 0.005, (meas, s['mad_sd'])
        assert abs(s['exkurt'] - 3.0) < 0.25, (meas, s['exkurt'])
        assert abs(s['skew']) < 0.05, (meas, s['skew'])
        assert s['ll_laplace'] > s['ll_normal']
        assert s['n'] == n
    # the demeaned measure removes the drift and the market term
    assert abs(stats(p['demeaned'])['mean']) < 1e-9


def test_member_growth_equals_log_delta_two_firms():
    """Hand-built two-firm unit, proportional sharing: the kernel's update is log(1 + delta_i) with
    delta_i = (1 - alpha) r_i + alpha sum_j w_j r_j - Phi(2), and growth_panel returns it as the member's growth."""
    alpha, phi2 = 0.3, 0.004
    log_states = np.array([np.log(3.0), np.log(1.0)])
    log_ret = np.array([0.05, -0.12])
    unit_members = np.array([[0, 1]], dtype=np.int32)
    out = process_conglomerate_pooling_numba(
        unit_members, np.array([2], dtype=np.int16), np.array([0], dtype=np.int64), log_ret, log_states,
        np.array([0.0, 0.0, phi2, 0.0]), alpha, 50, 100, 1, np.array([alpha]), False)
    new = out[0]
    w = np.array([0.75, 0.25])
    r = np.expm1(log_ret)
    delta = (1 - alpha) * r + alpha * (w @ r) - phi2
    np.testing.assert_allclose(new - log_states, np.log1p(delta), rtol=0, atol=1e-14)

    # the same two rows as a trace of log levels: one market per firm, both pooled in unit 0
    p = growth_panel(np.vstack([log_states, new]), np.zeros((2, 2), dtype=np.int16),
                     np.zeros((2, 1), dtype=bool), np.array([0, 1]), burn_in=0)
    assert p['member'].all() and (p['K'] == 2).all()
    np.testing.assert_allclose(p['raw'], np.log1p(delta), rtol=0, atol=1e-14)


def test_status_K_and_exclusions():
    """Pooled in step t if cong[t] or cong[t + 1] is a unit; lifts in step t or t - 1 are excluded."""
    T, F = 3, 4
    ls = np.zeros((T + 1, F))
    cong = np.array([[-1, -1, -1, -1],
                     [-1, 5, 5, -1],          # step 0: firms 1, 2 join unit 5
                     [3, 5, -1, -1],          # step 1: firm 2 leaves after pooling; firm 0 joins unit 3 ...
                     [3, 5, -1, -1]], dtype=np.int16)
    cong[2:, 3] = 3                           # ... with firm 3
    floor = np.zeros((F, T), dtype=bool)
    floor[2, 1] = True                        # firm 2 lifted in step 1 -> steps 1 and 2 excluded for firm 2
    p = growth_panel(ls, cong, floor, np.arange(F) // 2, burn_in=0)
    assert p['n_total'] == 12 and p['n_excluded'] == 2 and p['n_lift_end'] == 1 and p['n_lift_start_only'] == 1
    # kept firm-periods in (step, firm) order, firm 2 at steps 1 and 2 dropped
    expect_member = [False, True, True, False,     # step 0
                     True, True, True,             # step 1: firms 0, 1, 3 (2 dropped)
                     True, True, True]             # step 2: firms 0, 1, 3
    expect_K = [0, 2, 2, 0, 2, 2, 2, 2, 1, 2]   # step 2: unit 5 holds only firm 1
    assert p['member'].tolist() == expect_member
    assert p['K'].tolist() == expect_K


def test_demeaning_cancels_a_market_common_term():
    rng = np.random.default_rng(0)
    T, M, N = 30, 3, 10
    ls = rng.normal(size=(T + 1, M * N))
    shift = np.repeat(rng.normal(size=(T + 1, M)), N, axis=1)
    args = dict(cong=np.full((T + 1, M * N), -1, dtype=np.int16), floor=np.zeros((M * N, T), dtype=bool),
                home=np.repeat(np.arange(M), N), burn_in=5)
    a, b = growth_panel(ls, **args), growth_panel(ls + shift, **args)
    np.testing.assert_allclose(a['demeaned'], b['demeaned'], atol=1e-12)
    assert not np.allclose(a['raw'], b['raw'])


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'pilot_d', 'growth_distributions.csv')),
                    reason='outputs not generated')
def test_outputs():
    import pandas as pd
    d = pd.read_csv(os.path.join(ROOT, 'pilot_d', 'growth_distributions.csv'))
    s = d[d.table == 'status']
    assert len(s) == 9 * 2 * 4 and set(s.measure) == {'raw', 'demeaned', 'standardized', 'prelift'}
    # K bins partition the members up to firms whose pooled unit was not identifiable (K < 2)
    k = d[d.table == 'K'].groupby(['family', 'alpha', 'measure']).n.sum()
    m = s[s.status == 'member'].set_index(['family', 'alpha', 'measure']).n
    assert ((m - k.reindex(m.index)) / m).abs().max() < 1e-3
    dec = d[d.table == 'decile'].groupby(['family', 'alpha', 'measure', 'status']).n.sum()
    assert (dec == s.set_index(['family', 'alpha', 'measure', 'status']).n.reindex(dec.index)).all()
    assert os.path.exists(os.path.join(ROOT, 'figures', 'appendix', 'growth_density_alpha0.3.pdf'))


def test_prelift_and_reference_variance():
    """prelift adds the lifted periods back with the growth before the lift; with equal weights the reference
    variance is 1 - alpha (2 - alpha)(1 - 1/K)."""
    T, M, N, alpha = 4, 2, 3, 0.3
    F = M * N
    rng = np.random.default_rng(1)
    ls = rng.normal(size=(T + 1, F)) * 0.1
    ls[:, [0, 3, 1]] = 0.0                                     # firms 0, 3 (unit 7, equal weights) and 1 equal size
    cong = np.full((T + 1, F), -1, dtype=np.int16)
    cong[:, 0] = cong[:, 3] = 7
    floor = np.zeros((F, T), dtype=bool)
    floor[2, 1] = True
    jump = np.zeros((T + 1, F))
    jump[2, 2] = 0.5                                           # lift of firm 2 in step 1 (row 2)
    ls[2:, 2] += 0.5
    p = growth_panel(ls, cong, floor, np.arange(F) // N, burn_in=0, q_m=np.ones(M), jump=jump, alpha=alpha)
    a = p['all']
    assert len(a['prelift']) == T * F and len(p['raw']) == T * F - 2
    pre = (ls[1:] - ls[:-1] - jump[1:]).reshape(T, M, N)
    np.testing.assert_allclose(a['prelift'], (pre - pre.mean(axis=2, keepdims=True)).ravel(), atol=1e-12)
    ref = a['ref_var'].reshape(T, F)
    np.testing.assert_allclose(ref[:, [0, 3]], 1 - alpha * (2 - alpha) * (1 - 1 / 2), atol=1e-12)
    assert (ref[:, [1, 2, 4, 5]] == 1).all()
