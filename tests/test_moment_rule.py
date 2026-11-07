"""C36: decision_rule='moment' (collaborative_growth.member_gain_moment and its use in the kernel, exit test and
alpha renegotiation). The identity of every existing scenario (brief test e) is checked at full T by
analytics/identity_c35.py --same-schema on the Festus reruns (diagnostics/identity_c36.csv)."""
import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW


def gains(rule, returns, sizes, alpha, Phi, share=1, g=0.0):
    """Every member's estimated gain in the set of all columns (rule 1 = loggain, 2 = moment)."""
    K = returns.shape[1]
    members = np.arange(K, dtype=np.int64)
    eps, sz, ss, sse = cg.prepare_window_loggain(members, returns, sizes, g)
    su, su2 = cg.prepare_rule_sums(rule, eps, sz, g)
    return np.array([cg.rule_gain_from_prepared(rule, i, K, eps, sz, ss, sse, su, su2, alpha, Phi, share, g)
                     for i in range(K)])


HAND_R = np.array([[0.1, 0.0], [-0.1, 0.2], [0.0, -0.2]])   # h = 3 periods x 2 members, both of mean 0


def test_hand_two_members():
    """(a) Two members, equal constant sizes, alpha = 0.5, window of 3 periods, returns HAND_R (mean 0).
    u0 = (0.1, -0.1, 0), u1 = (0, 0.2, -0.2); V(u0) = 0.01, V(u1) = 0.04 (divisor h - 1 = 2).
    y0 = 0.75 u0 + 0.25 u1 = (0.075, -0.025, -0.05), V(y0) = 0.004375;
    y1 = 0.25 u0 + 0.75 u1 = (0.025, 0.125, -0.15), V(y1) = 0.019375.
    Transfer term: mean over periods of (u0^2 + u1^2)/2 with divisor 2 = (0.005 + 0.025 + 0.02)/2 = 0.025.
    D0 = 1/2 [0.5 * 0.01 + 0.5 * 0.025 - 0.004375] = 0.0065625
    D1 = 1/2 [0.5 * 0.04 + 0.5 * 0.025 - 0.019375] = 0.0065625  (equal: both end with the same pooled path's
    second moments; 1/2 [V(u_i) - V(y_i)] alone would give 0.0028125 and 0.0103125).
    With Phi = 0.001 and g = 0: A = 0.999, r = 1/0.999, D_i = log(0.999) + 1/2 (r Q_i - r^2 V(y_i)) with
    Q0 = 0.0175, Q1 = 0.0325. The same holds under equal sharing (equal sizes: c_j = 1/2) and for any g
    (u = eps - g is unchanged and A = e^g)."""
    sizes = np.ones((3, 2))
    for share in (0, 1):
        for g in (0.0, 0.055):
            d = gains(2, HAND_R, sizes, 0.5, 0.0, share, g)
            assert np.allclose(d, [0.0065625, 0.0065625], atol=1e-15, rtol=1e-12), (share, g, d)
        d = gains(2, HAND_R, sizes, 0.5, 0.001, share, 0.0)
        r = 1 / 0.999
        expect = [np.log(0.999) + 0.5 * (r * q - r * r * v) for q, v in ((0.0175, 0.004375), (0.0325, 0.019375))]
        assert np.allclose(d, expect, atol=1e-15, rtol=1e-12), (share, d, expect)


def test_short_window_and_ruin_give_minus_inf():
    """h < 3 gives -inf; so does a period whose replayed p is <= -1 (as under loggain)."""
    sizes = np.ones((2, 2))
    assert np.all(gains(2, HAND_R[:2], sizes, 0.5, 0.0) == -np.inf)
    r = HAND_R.copy()
    r[1, 0] = -40.0                              # e^-40 - 1 ~ -1: (1 - a) e_0 + a mean e <= -1 with Phi > 0
    d = gains(2, r, np.ones((3, 2)), 0.0, 0.01)
    assert d[0] == -np.inf and gains(1, r, np.ones((3, 2)), 0.0, 0.01)[0] == -np.inf


@pytest.fixture(scope='module')
def iid_normal():
    rng = np.random.default_rng(1)
    return rng.normal(0.0, 0.2, size=(200_000, 10))


def section2(alpha, sigma2, K):
    return 0.5 * alpha * (2 - alpha) * sigma2 * (1 - 1 / K)


def test_symmetric_increasing_in_alpha_and_K(iid_normal):
    """(b) Symmetric i.i.d. members (sigma = 0.2), constant sizes, Phi = 0, long window (200,000 periods):
    member 0's Delta-tilde increases in alpha (0.1..0.9 at K = 5) and in K (2..10 at alpha = 0.3), and is within
    3% of 1/2 a (2 - a) sigma^2 (1 - 1/K) at the window's own variance."""
    R = iid_normal
    s2 = R.var(axis=0, ddof=1).mean()
    alphas = np.arange(0.1, 0.95, 0.1)
    d_a = np.array([gains(2, R[:, :5], np.ones((len(R), 5)), a, 0.0)[0] for a in alphas])
    assert np.all(np.diff(d_a) > 0)
    assert np.allclose(d_a, [section2(a, s2, 5) for a in alphas], rtol=0.03)
    d_k = np.array([gains(2, R[:, :k], np.ones((len(R), k)), 0.3, 0.0)[0] for k in range(2, 11)])
    assert np.all(np.diff(d_k) > 0)
    assert np.allclose(d_k, [section2(0.3, s2, k) for k in range(2, 11)], rtol=0.03)


def test_marginal_member_converges_to_section2(iid_normal):
    """(c) The marginal gain of the K-th member for an existing member, Delta-tilde_0(K) - Delta-tilde_0(K - 1),
    is within 5% of 1/2 a (2 - a) sigma^2 [1/(K-1) - 1/K] for K = 3..10 (alpha = 0.3, 200,000 periods; the
    differences use the same draws, so their sampling error is far below 5%)."""
    R = iid_normal
    s2 = R.var(axis=0, ddof=1).mean()
    d = {k: gains(2, R[:, :k], np.ones((len(R), k)), 0.3, 0.0)[0] for k in range(2, 11)}
    for k in range(3, 11):
        expect = 0.5 * 0.3 * 1.7 * s2 * (1 / (k - 1) - 1 / k)
        assert abs((d[k] - d[k - 1]) / expect - 1) < 0.05, (k, d[k] - d[k - 1], expect)


@pytest.mark.parametrize('share', [1, 0])
def test_heterogeneous_members_match_loggain_on_long_window(share):
    """Members of different volatility (sigma 0.1, 0.3, 0.2) and size (1, 2, 0.5), alpha = 0.3, Phi = 1e-4,
    g = 0.055, 200,000 periods: Delta-tilde is within 10% of loggain's Delta-hat for every member under both
    sharing rules, i.e. it estimates the same objective. 1/2 [V(u_i) - V(y_i)] without the transfer term does
    not (it misses the low-volatility member's gain by more than half).
    Catches: a dropped or mis-signed transfer term, or the equal-split level term."""
    rng = np.random.default_rng(7)
    sig = np.array([0.1, 0.3, 0.2])
    R = rng.normal(0.0, 1.0, size=(200_000, 3)) * sig
    sizes = np.tile([1.0, 2.0, 0.5], (len(R), 1))
    dl = gains(1, R, sizes, 0.3, 1e-4, share, 0.055)
    dm = gains(2, R, sizes, 0.3, 1e-4, share, 0.055)
    assert np.all(np.abs(dm / dl - 1) < 0.10), (dl, dm)
    if share == 1:
        u = R - R.mean(axis=0)
        w = sizes[0] / sizes[0].sum()
        y0 = 0.7 * u[:, 0] + 0.3 * (u @ w)
        literal = 0.5 * (u[:, 0].var(ddof=1) - y0.var(ddof=1)) - 1e-4
        assert abs(literal / dl[0] - 1) > 0.5


def test_noisy_window_loggain_rejects_moment_accepts():
    """(d) A standalone joins a unit of two (alpha = 0.3, Phi = 0, i.i.d. normal sigma = 0.2, sizes a random walk
    driven by the returns, 50-period window). The true marginal gain of the 3rd member for the incumbents is
    1/2 a (2 - a) sigma^2 (1/2 - 1/3) = 1.7e-3 > 0 (and the newcomer's is larger). Over 400 windows, at least one
    is rejected by loggain (an incumbent's Delta-hat(3) <= Delta-hat(2)) and accepted by moment, and moment
    rejects less often than loggain."""
    rng = np.random.default_rng(3)
    rej_l = rej_m = 0
    found = None
    for k in range(400):
        R = rng.normal(0.0, 0.2, size=(50, 3))
        sizes = np.exp(np.cumsum(np.vstack([np.zeros(3), R[:-1]]), axis=0))
        ok = {}
        for rule in (1, 2):
            d3 = gains(rule, R, sizes, 0.3, 0.0, 1, 0.055)
            d2 = gains(rule, R[:, :2], sizes[:, :2], 0.3, 0.0, 1, 0.055)
            ok[rule] = bool(np.all(d3[:2] > np.maximum(0.0, d2)) and d3[2] > 0)
        rej_l += not ok[1]
        rej_m += not ok[2]
        if found is None and not ok[1] and ok[2]:
            found = k
    assert found is not None
    assert rej_m < rej_l, (rej_m, rej_l)


def test_default_rule_code_is_loggain():
    """find_optimal_alpha's new rule_code argument defaults to loggain (existing callers unchanged)."""
    rng = np.random.default_rng(5)
    R = rng.normal(0, 0.2, size=(50, 3))
    S = np.exp(np.cumsum(R, axis=0))
    m = np.arange(3, dtype=np.int32)
    assert cg.find_optimal_alpha(m, R, S, 0.1, 1e-4, 1, 0.055) == cg.find_optimal_alpha(m, R, S, 0.1, 1e-4, 1, 0.055, 1)


def test_moment_model_runs_and_requires_g():
    """A short run under the moment rule forms units, records the binding-gain histogram, renegotiates alpha,
    and differs from the loggain run with the same seed; decision_rule='moment' without g raises."""
    hp_m = cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, decision_rule='moment', alpha_endogenous=True))[-1]
    hp_l = cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, decision_rule='loggain', alpha_endogenous=True))[-1]
    assert hp_m['mergers_by_K'].sum() > 0
    assert np.array_equal(hp_m['binding_gain_hist_by_K'].sum(1), hp_m['proposals_by_K'])
    assert not np.array_equal(hp_m['final_log_states'], hp_l['final_log_states'])
    assert np.nanstd(hp_m['alpha_history']) > 0
    with pytest.raises(ValueError):
        cg.model(**PARAMS, **dict(KW, decision_rule='moment', g=None))
