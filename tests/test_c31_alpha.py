"""C31: endogenous alpha. Decisions at the unit's own alpha, renegotiation every period, alpha of merged
and newly allocated units.

One seeded M=N=20, T=600 run with alpha_endogenous=True (otherwise the configuration of
tests/test_floor_exit.py). Instrumentation: model() looks up merger_kernel_numba and
compute_exit_candidates_numba as module globals, so the tests wrap them (no debug hook in the model):
the kernel wrapper snapshots cong_alpha, cong_size, cong_active and firm_conglom before and after every
call; the exit wrapper recomputes each candidate's Delta_current by hand.
"""
import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW

LOOKBACK = PARAMS[9]
SHARE = PARAMS[3]


def _gain_current(firm, alpha, a):
    """Delta_current of `firm` in its unit at `alpha`, recomputed from the exit test's own arguments."""
    (_, states_buf, returns_buf, _, lookback, step, _, g, _, sharing_code, costs, firm_conglom,
     cong_firms, cong_size, _) = a
    cid = firm_conglom[firm]
    members = cong_firms[cid, :cong_size[cid]].copy()
    j = int(np.nonzero(members == firm)[0][0])
    past_states = cg.get_historical_states_numba(states_buf, step - lookback, step, members)
    past_returns = cg.get_historical_returns_numba(returns_buf, step - lookback, step, members)
    eps, sz, sum_s, sum_s_eps = cg.prepare_window_loggain(members, past_returns, np.exp(past_states), g)
    return cg.member_gain_from_prepared(j, len(members), eps, sz, sum_s, sum_s_eps, alpha,
                                        costs[len(members)], sharing_code)


@pytest.fixture(scope='module')
def run():
    kernel_calls, exit_checks = [], []
    orig_kernel, orig_exit = cg.merger_kernel_numba, cg.compute_exit_candidates_numba

    def kernel(*a):
        step, firm_conglom, cong_size, cong_active, cong_alpha = a[0], a[5], a[9], a[10], a[23]
        before = (cong_alpha.copy(), cong_size.copy(), cong_active.copy(), firm_conglom.copy())
        out = orig_kernel(*a)
        after = (cong_alpha.copy(), cong_size.copy(), cong_active.copy(), firm_conglom.copy())
        kernel_calls.append((int(step), before, after))
        return out

    def exit_test(*a):
        out = orig_exit(*a)
        firm_conglom, cong_alpha = a[11], a[14]
        exited = set(int(f) for f in out)
        for f in a[3]:
            alpha_unit = cong_alpha[firm_conglom[f]]
            exit_checks.append((int(a[5]), int(f), float(alpha_unit), _gain_current(f, alpha_unit, a),
                                _gain_current(f, SHARE, a), int(f) in exited))
        return out

    cg.merger_kernel_numba, cg.compute_exit_candidates_numba = kernel, exit_test
    try:
        cg.model(PARAMS, **dict(KW, alpha_endogenous=True))
    finally:
        cg.merger_kernel_numba, cg.compute_exit_candidates_numba = orig_kernel, orig_exit
    return kernel_calls, exit_checks


def test_alpha_renegotiated_every_period(run):
    """cong_alpha of some active unit changes between the previous kernel call and the current one (where only
    the alpha update runs) at a step that is not a multiple of lookback.
    Catches: renegotiation still restricted to step % lookback == 0 (Step 1b not applied)."""
    calls, _ = run
    off_grid = 0
    for (_, _, prev_after), (step, before, _) in zip(calls[:-1], calls[1:]):
        changed = (before[0] != prev_after[0]) & before[2]
        if changed.any() and step % LOOKBACK != 0:
            off_grid += 1
    assert off_grid > 0


def test_exit_test_uses_unit_alpha(run):
    """For every exit candidate the decision equals Delta_current(cong_alpha[unit]) < 0, recomputed by hand
    from the exit test's arguments, and in at least one case that decision differs from the one at `share`.
    Catches: the exit test still evaluating Delta_current at the global share."""
    _, checks = run
    assert len(checks) > 0
    for step, firm, alpha_unit, g_unit, g_share, exited in checks:
        assert exited == (g_unit < 0), (step, firm)
    assert any((g_unit < 0) != (g_share < 0) for *_, g_unit, g_share, _ in checks)


def _unit_unit_mergers(calls):
    """(alpha_after_target, alpha_init, alpha_target, size_init, size_target) for unit+unit mergers detected
    unambiguously: a unit active before and inactive after whose members (together with all members of one
    other pre-existing unit, and no one else) now form that other unit."""
    found = []
    for _, (a0, s0, act0, fc0), (a1, s1, act1, fc1) in calls:
        for init in np.nonzero(act0 & ~act1)[0]:
            init_members = np.nonzero(fc0 == init)[0]
            targets = set(fc1[init_members]) - {-1}
            if len(targets) != 1:
                continue
            targ = targets.pop()
            if not act0[targ] or targ == init:
                continue
            union = np.union1d(init_members, np.nonzero(fc0 == targ)[0])
            if np.array_equal(np.sort(np.nonzero(fc1 == targ)[0]), union):
                found.append((a1[targ], a0[init], a0[targ], int(s0[init]), int(s0[targ])))
    return found


def test_unit_unit_merger_takes_larger_units_alpha(run):
    """After a unit+unit merger cong_alpha[target] equals the alpha of the larger unit (initiator on ties),
    and at least one such merger joins units with different alphas and sizes (so the rule is tested).
    Catches: the merged unit keeping the target's alpha, taking the initiator's, or resetting to share."""
    calls, _ = run
    mergers = _unit_unit_mergers(calls)
    assert len(mergers) > 0
    for a_after, a_init, a_targ, s_init, s_targ in mergers:
        assert a_after == (a_init if s_init >= s_targ else a_targ)
    assert any(a_init != a_targ and s_init != s_targ for _, a_init, a_targ, s_init, s_targ in mergers)


def test_new_slot_gets_share(run):
    """A slot that becomes active in a kernel call has cong_alpha == share, including reused slots that held
    a different (stale) alpha before. Catches: new solo+solo units inheriting a stale alpha."""
    calls, _ = run
    stale_reused = 0
    for _, (a0, _, act0, _), (a1, s1, act1, _) in calls:
        # only slots that end the call as two-firm units: a later unit+unit merger in the same call
        # (size >= 4) legitimately sets the larger unit's alpha
        for cid in np.nonzero(~act0 & act1 & (s1 == 2))[0]:
            assert a1[cid] == SHARE, cid
            stale_reused += a0[cid] != SHARE
    assert stale_reused > 0
