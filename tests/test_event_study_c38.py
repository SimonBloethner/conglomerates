"""C38 Task 1: alternative matchings of the event study (analysis/event_study.py) and the pre/post decomposition.

Synthetic market 0 with a joiner (firm 0, enters at t = 50, window l = 20) and three standalone candidates whose
raw share, owner-path share and recapitalisation history are set so that each matching picks a different control:
  firm  logshare[t]  floor hits in [t-l, t) (step: jump)   owner key = logshare[t] - window jumps
  0     -2.00        1 (39: 0.3)                            -2.30
  1     -2.02        0                                      -2.02   raw nearest (gap 0.02)
  2     -1.90        1 (34: 0.4)                            -2.30   owner nearest (gap 0)
  3     -2.08        1 (44: 0.3)                            -2.38   nearest with one hit (gap 0.08)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.event_study import events_from_trace, summarize  # noqa: E402

T, L, ENTRY = 120, 20, 50


def trace():
    F = 8
    home = np.repeat(np.arange(2), 4)
    logshare = np.full((T + 1, F), -np.log(4.0))
    unit = np.full((T + 1, F), -1, dtype=np.int16)
    floor = np.zeros((F, T), dtype=bool)
    jump = np.zeros((T + 1, F))
    unit[ENTRY:, 0] = 0
    unit[ENTRY:, 4] = 0                      # its partner in market 1 (also enters at t: an event without controls)
    for i, level, hits in ((0, -2.00, [(39, 0.3)]), (1, -2.02, []), (2, -1.90, [(34, 0.4)]),
                           (3, -2.08, [(44, 0.3)])):
        logshare[:, i] = level
        for step, j in hits:
            floor[i, step] = True
            jump[step + 1, i] = j
    return logshare, unit, floor, home, jump


def event(**kw):
    ls, unit, floor, home, jump = trace()
    ev = events_from_trace(ls, unit, floor, home, l=L, tol=0.25, jump=jump, **kw)
    return ev[(ev.firm == 0) & (ev.step == ENTRY)]


def test_raw_is_default_and_picks_raw_nearest():
    """match='raw' (the default) keeps the C29-C37 control: nearest raw log share."""
    e = event()
    assert len(e) == 1 and e.control.iloc[0] == 1
    assert event(match='raw').equals(e)
    assert e.n_hits_before.iloc[0] == 1 and e.control_n_hits_before.iloc[0] == 0


def test_owner_matching_nets_out_window_recap():
    """match='owner' compares logshare[t] - (jumps in (t-l, t]); firm 2's raw gap is 0.10 but owner gap 0.
    Catches: the key built from the cumulative jump since step 0 rather than over the window."""
    e = event(match='owner')
    assert e.control.iloc[0] == 2


def test_recap_matching_requires_equal_hit_count():
    """match='recap': among candidates with the joiner's number of pre-window hits (1), nearest raw share."""
    e = event(match='recap')
    assert e.control.iloc[0] == 3 and e.control_n_hits_before.iloc[0] == e.n_hits_before.iloc[0] == 1


def test_hitfree_window():
    """hitfree_k=10: the joiner's hit (step 39) is outside [40, 50), so it survives, and firm 3 (hit at 44) is
    no candidate; hitfree_k=20: the joiner hit within [30, 50), so the event is dropped."""
    e = event(hitfree_k=10)
    assert len(e) == 1 and e.control.iloc[0] == 1
    e = event(hitfree_k=10, match='recap')     # one-hit candidates: 2 (hit at 34, outside) survives, 3 does not
    assert e.control.iloc[0] == 2
    assert len(event(hitfree_k=20)) == 0


def test_owner_matching_needs_jump():
    ls, unit, floor, home, _ = trace()
    with pytest.raises(AssertionError):
        events_from_trace(ls, unit, floor, home, l=L, match='owner')


def test_decomposition_is_additive():
    """did_mean = post_gap_mean - pre_gap_mean on both paths, and the control owner columns are those of the
    matched control. Random walk trace with floor jumps."""
    rng = np.random.default_rng(0)
    F, Tr = 40, 400
    home = np.repeat(np.arange(4), 10)
    jump = np.where(rng.random((Tr + 1, F)) < 0.03, rng.random((Tr + 1, F)) * 0.3, 0.0)
    jump[0] = 0
    ls = np.cumsum(rng.normal(0, 0.05, (Tr + 1, F)) + jump, axis=0) - 2.0
    floor = (jump[1:] > 0).T
    unit = np.full((Tr + 1, F), -1, dtype=np.int16)
    for k, (t, i) in enumerate([(100, 0), (150, 11), (200, 22), (250, 33), (120, 5), (300, 15)]):
        unit[t:, i] = k
    ev = events_from_trace(ls, unit, floor, home, l=50, tol=5.0, jump=jump)
    s = summarize(ev)
    m = ev[ev.control >= 0]
    assert len(m) == 6
    assert np.isclose(m.did.mean(), s['post_gap_mean'] - s['pre_gap_mean'], atol=1e-12)
    assert np.isclose(s['did_owner_mean'], s['post_gap_owner_mean'] - s['pre_gap_owner_mean'], atol=1e-12)
    owner = ls - np.cumsum(jump, axis=0)
    r = m.iloc[0]
    c, t = int(r.control), int(r.step)
    assert np.isclose(r.control_before_owner, (owner[t, c] - owner[t - 50, c]) / 50)
    assert np.isclose(r.control_after_owner, (owner[t + 50, c] - owner[t, c]) / 50)
    assert np.isclose(s['control_after_owner_m_mean'] - s['joiner_after_owner_m_mean'], -s['post_gap_owner_mean'])
