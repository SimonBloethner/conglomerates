"""C35 Task 2: proposals and mergers by K = size of the merged set, and the binding-gain histogram."""
import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW



@pytest.fixture(scope='module')
def run():
    return cg.model(**dict(PARAMS, search_prob=0.3), **KW)


def test_by_K_sums_equal_type_counts_per_period(run):
    """Per period: proposals_by_K[2] == proposals_ss, sum over K >= 3 == proposals_sc + proposals_cc (and the
    same for mergers); K = 0, 1 never counted; the post-burn-in sums are the stored totals.
    Catches: a counter incremented at a different point from the per-type ones, or K off by one."""
    hp = run[-1]
    P, Mg = hp['proposals_by_K_per_period'], hp['mergers_by_K_per_period']
    assert P.shape == (PARAMS['T'], PARAMS['M'] + 1)
    assert np.array_equal(P[:, 2], hp['proposals_ss_per_period'])
    assert np.array_equal(P[:, 3:].sum(1), hp['proposals_sc_per_period'] + hp['proposals_cc_per_period'])
    assert np.array_equal(Mg[:, 2], hp['mergers_ss_per_period'])
    assert np.array_equal(Mg[:, 3:].sum(1), hp['mergers_sc_per_period'] + hp['mergers_cc_per_period'])
    assert np.array_equal(P.sum(1), run[8]) and np.array_equal(Mg.sum(1), run[7])
    assert not P[:, :2].any() and not Mg[:, :2].any()
    b = KW['burn_in']
    assert np.array_equal(hp['proposals_by_K'], P[b:].sum(0))
    assert np.array_equal(hp['mergers_by_K'], Mg[b:].sum(0))
    assert (P[:, 3:].sum() > 0) and (Mg[:, 3:].sum() > 0)     # the run has solo+unit / unit+unit proposals


def test_acceptance_by_K_summary(run):
    """summary['acceptance_by_K'] = mergers / proposals by K after burn-in, NaN where no proposals."""
    hp = run[-1]
    acc, P, Mg = hp['summary']['acceptance_by_K'], hp['proposals_by_K'], hp['mergers_by_K']
    assert len(acc) == PARAMS['M'] + 1
    assert np.all(np.isnan(acc[P == 0]))
    assert np.allclose(acc[P > 0], Mg[P > 0] / P[P > 0])


def test_binding_hist_counts_and_sign(run):
    """One histogram count per post-burn-in proposal, in the row of its K; the proposals with a binding gain
    in a bin above 0 are the accepted ones (a two-solo merger can still fail for want of a free slot, never
    the case here), up to proposals within one bin of 0.
    Catches: a histogram filled before burn-in, or a binding gain with the wrong sign."""
    hp = run[-1]
    H, edges = hp['binding_gain_hist_by_K'], hp['binding_gain_edges']
    assert H.shape == (PARAMS['M'] + 1, cg.BINDING_GAIN_BINS) and len(edges) == cg.BINDING_GAIN_BINS + 1
    assert np.array_equal(H.sum(1), hp['proposals_by_K'])
    zero = int(np.argmin(np.abs(edges)))
    above = H[:, zero:].sum(1)
    assert np.all(above >= hp['mergers_by_K'])
    assert np.all(above - hp['mergers_by_K'] <= H[:, zero])


def test_two_firm_economy_only_K2():
    """M = 2 markets: a unit of two is full, so every proposal is solo+solo, K = 2 only."""
    hp = cg.model(**dict(PARAMS, M=2, N=2, T=400, search_prob=0.5), **dict(KW, burn_in=50))[-1]
    P, Mg = hp['proposals_by_K_per_period'], hp['mergers_by_K_per_period']
    assert P.shape[1] == 3
    assert P[:, 2].sum() > 0
    assert not P[:, :2].any() and not Mg[:, :2].any()
    assert np.array_equal(P[:, 2], hp['proposals_ss_per_period'])
    assert not hp['proposals_sc_per_period'].any() and not hp['proposals_cc_per_period'].any()


def test_recording_does_not_change_the_run():
    """The member loop runs to the end when the histogram is recorded and stops at the first rejecting member
    otherwise; the two give the same run (same decisions, same RNG draws).
    Catches: the binding-gain loop changing an accept decision or consuming random numbers."""
    orig = cg.merger_kernel_numba

    def never_record(*a):
        return orig(*a[:-1], False)

    ref = cg.model(**dict(PARAMS, search_prob=0.3), **KW)
    cg.merger_kernel_numba = never_record
    try:
        alt = cg.model(**dict(PARAMS, search_prob=0.3), **KW)
    finally:
        cg.merger_kernel_numba = orig
    for i in (0, 1, 7, 8, 9):
        assert np.array_equal(ref[i], alt[i])
    for k in ('final_log_states', 'floor_hits', 'final_firm_conglom', 'proposals_by_K', 'mergers_by_K'):
        assert np.array_equal(ref[-1][k], alt[-1][k]), k
    assert not alt[-1]['binding_gain_hist_by_K'].any()
