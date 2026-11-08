"""C39 Task 2: decision_rule='cv', the loggain replay with its first-order term removed (member_gain_cv)."""
import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW


def prepared(R, S, g):
    K = R.shape[1]
    eps, sz, sum_s, sum_s_eps = cg.prepare_window_loggain(np.arange(K), R, S, g)
    sum_su, sum_su2 = cg.prepare_window_moment(eps, sz, g)
    return eps, sz, sum_s, sum_s_eps, sum_su, sum_su2


def three(R, S, a, phi, sharing, g, i=0):
    """(loggain, moment, cv) estimates of member i."""
    K = R.shape[1]
    eps, sz, s1, s2, su, su2 = prepared(R, S, g)
    return (cg.member_gain_from_prepared(i, K, eps, sz, s1, s2, a, phi, sharing),
            cg.member_gain_moment(i, K, eps, sz, s1, s2, su, su2, a, phi, sharing, g),
            cg.member_gain_cv(i, K, eps, sz, s1, s2, su, a, phi, sharing, g))


def cv_by_hand(R, S, a, phi, sharing, g, i):
    """Direct transcription of the definition: demean, replay, subtract r_t a (sum_j c_jt u_jt - u_it)."""
    h, K = R.shape
    u = R - R.mean(0)                       # u = eps - g
    eps = u + g
    e = np.expm1(eps)
    total = 0.0
    for t in range(h):
        s = S[t]
        if sharing == 1:
            c = s / s.sum()
            p = (1 - a) * e[t, i] + a * (s * e[t]).sum() / s.sum() - phi
        else:
            c = s / (K * s[i])
            p = (1 - a) * e[t, i] + (a * (s * e[t]).sum() - phi * s.sum()) / K / s[i]
        ct = c.sum()
        A = np.exp(g) * (1 - a + a * ct) + a * (1 - ct) - phi * ct
        r = np.exp(g) / A
        total += np.log1p(p) - np.log1p(e[t, i]) - r * a * ((c * u[t]).sum() - u[t, i])
    return total / h


@pytest.mark.parametrize('sharing', [1, 0])
def test_two_member_case_by_hand(sharing):
    """(a) h = 4, K = 2, sizes changing within the window, both sharing rules, both members."""
    R = np.array([[0.10, -0.05], [-0.02, 0.07], [0.04, 0.01], [-0.08, 0.03]]) + 0.055
    S = np.array([[1.0, 2.0], [1.1, 1.9], [1.05, 2.1], [1.08, 2.15]])
    for i in (0, 1):
        _, _, d_cv = three(R, S, 0.3, 2e-4, sharing, 0.055, i)
        assert abs(d_cv - cv_by_hand(R, S, 0.3, 2e-4, sharing, 0.055, i)) < 1e-14


def test_minus_inf_and_short_window():
    """Same -inf rule as loggain / moment: h < 3, or a replayed p <= -1."""
    R = np.full((2, 2), 0.05)
    S = np.ones((2, 2))
    assert three(R, S, 0.3, 0.0, 1, 0.055)[2] == -np.inf
    R = np.array([[0.05, 0.05], [-200.0, -200.0], [0.05, 0.05], [0.05, 0.05]])   # both collapse: p = -1
    S = np.ones((4, 2))
    assert three(R, S, 0.3, 0.0, 1, 0.055)[2] == -np.inf


def test_exact_identities_on_random_windows():
    """(b) On random 50-period windows with sizes that follow the returns: cv - loggain is exactly minus the sample
    mean of the subtracted first-order term; with the sizes frozen the term sums to zero, so cv = loggain."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        K = int(rng.integers(2, 9))
        R = rng.normal(0.055, rng.uniform(0.05, 0.25), (50, K))
        S = np.exp(np.cumsum(np.vstack([rng.normal(0, 0.5, (1, K)), R[:-1] - 0.055]), axis=0))
        a = rng.uniform(0.05, 0.9)
        d_hat, _, d_cv = three(R, S, a, 3e-5, 1, 0.055)
        fo = d_hat - cv_by_hand(R, S, a, 3e-5, 1, 0.055, 0)
        assert abs((d_hat - d_cv) - fo) < 1e-12
        S0 = np.tile(S[0], (50, 1))
        f_hat, _, f_cv = three(R, S0, a, 3e-5, 1, 0.055)
        assert abs(f_hat - f_cv) < 1e-12


def test_long_window_convergence():
    """(c) Symmetric i.i.d. members with sizes that follow their own returns and revert (AR(1) in logs): with
    h = 200,000 the three rules agree within 2% of the gain, and with the analytical gain
    1/2 sigma^2 [1 - (1-a)^2 - 2a(1-a) c_i - a^2 sum c^2], averaged over the window's weights, within 3%."""
    rng = np.random.default_rng(2)
    h, K, sig, a, g, phi = 200_000, 4, 0.1, 0.3, 0.055, 0.0
    R = rng.normal(g, sig, (h, K))
    logs = np.zeros((h, K))
    for t in range(1, h):
        logs[t] = 0.95 * logs[t - 1] + (R[t - 1] - g)
    S = np.exp(logs)
    d_hat, d_mom, d_cv = three(R, S, a, phi, 1, g)
    c = S / S.sum(1, keepdims=True)
    analytic = 0.5 * sig ** 2 * np.mean(1 - (1 - a) ** 2 - 2 * a * (1 - a) * c[:, 0] - a * a * (c ** 2).sum(1))
    for d in (d_hat, d_mom):
        assert abs(d - d_cv) < 0.02 * analytic
    assert abs(d_cv - analytic) < 0.03 * analytic


def test_model_runs_under_cv():
    """The model runs end to end under decision_rule='cv' (consent, exit review, alpha renegotiation), forms units,
    and differs from the moment run on the same seed."""
    hp_cv = cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, decision_rule='cv', alpha_endogenous=True))[-1]
    hp_m = cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, decision_rule='moment', alpha_endogenous=True))[-1]
    assert hp_cv['mergers_by_K'].sum() > 0
    assert not np.array_equal(hp_cv['final_log_states'], hp_m['final_log_states'])
    with pytest.raises(ValueError):
        cg.model(**PARAMS, **dict(KW, decision_rule='cv', g=None))
