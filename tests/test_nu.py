"""C37: the nu block. Student-t with nu = 2.5 (finite variance, infinite kurtosis) under the moment rule.

- The moment rule's gain stays finite (or -inf by the ruin rule) for windows that hold the extreme draws a t(2.5)
  produces at the pilot's scale; never NaN or +inf.
- A short model run with family 'student_t' and nu = 2.5 under the moment rule has finite states and metrics, and
  differs from nu = 4 with the same seed (nu reaches the shock draw).
- run_pilot_c.run_scenario passes the scenario's nu."""
import numpy as np
import pytest
from scipy import stats

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW
from tests.test_moment_rule import gains


def test_moment_gain_finite_with_extreme_draws():
    """Windows of 50 periods x 5 members, t(2.5) shocks at IQR 0.3 relative to a market, with one draw replaced by
    an extreme value of either sign (|eps| up to 800 in log units, the largest a t(2.5) reaches over ~1e8 draws at
    this scale is ~250). Every member's gain is finite or -inf (ruin rule), never NaN or +inf, under both sharing
    rules and with and without cost."""
    rng = np.random.default_rng(25)
    s = 0.3 / (2 * stats.t.ppf(0.75, df=2.5))
    for k in range(200):
        R = s * rng.standard_t(2.5, size=(50, 5))
        R[rng.integers(50), rng.integers(5)] = rng.choice([-800.0, -50.0, 20.0, 250.0])
        sizes = np.exp(np.clip(np.cumsum(np.vstack([np.zeros(5), R[:-1]]), axis=0), -300, 300))
        for share in (0, 1):
            for phi in (0.0, 1e-4):
                d = gains(2, R, sizes, 0.3, phi, share, 0.055)
                assert not np.isnan(d).any() and not (d == np.inf).any(), (k, share, phi, d)


@pytest.mark.parametrize('nu', [2.5, 4.0])
def test_short_run_student_t_moment(nu):
    hp = cg.model(**dict(PARAMS, search_prob=0.3),
                  **dict(KW, family='student_t', nu=nu, decision_rule='moment', seed=7))[-1]
    assert np.isfinite(hp['final_log_states']).all()
    summ = hp['summary']
    for k in ('K_mean', 'floor_hit_rate_member', 'floor_hit_rate_standalone', 'mergers_per_period'):
        assert np.isfinite(summ[k]), k
    assert np.array_equal(hp['binding_gain_hist_by_K'].sum(1), hp['proposals_by_K'])
    assert hp['mergers_by_K'].sum() > 0


def test_nu_reaches_the_draws():
    a = cg.model(**PARAMS, **dict(KW, family='student_t', nu=2.5, decision_rule='moment', seed=7))[-1]
    b = cg.model(**PARAMS, **dict(KW, family='student_t', nu=4.0, decision_rule='moment', seed=7))[-1]
    assert not np.array_equal(a['final_log_states'], b['final_log_states'])


def test_run_scenario_passes_nu(monkeypatch):
    import run_pilot_c
    seen = {}

    def fake_model(**kw):
        seen.update(kw)
        raise StopIteration

    monkeypatch.setattr(run_pilot_c, 'model', fake_model)
    base = dict(proportional=False, M=2, N=2, T=10, alpha=0.1, merge_thresh=0.05, lookback=5,
                cost_type='power_law', c0=1e-5, c1=1.2, c2=1e-3, seed=1, growth_process='log_family',
                log_family='student_t', mu_range=[0.01, 0.1], sigma_range=[0.1, 0.3], floor_c=0.1, cross_corr=0.0,
                metric_every=5, sharing_rule='proportional', burn_in=5, g=0.055, market_size_fixed=True,
                decision_rule='moment')
    for nu, expect in ((2.5, 2.5), (None, 3.0)):
        with pytest.raises(StopIteration):
            run_pilot_c.run_scenario(dict(base, nu=nu))
        assert seen['nu'] == expect
    with pytest.raises(StopIteration):
        run_pilot_c.run_scenario(base)              # pilot C scenarios have no nu key
    assert seen['nu'] == 3.0
