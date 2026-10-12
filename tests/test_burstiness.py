"""C32: floor-hit clustering (analysis/burstiness.py) on synthetic traces with known dispersion.

Synthetic traces: F = 200 firms in 4 markets (N = 50), T = 5000 periods, burn-in 500.
"""
import numpy as np
from scipy.stats import binom

from analysis.burstiness import burstiness

F, T, M, BURN_IN = 200, 5000, 4, 500
HOME = np.repeat(np.arange(M), F // M)


def test_independent_hazard_is_binomial():
    """Independent firms, constant hazard 0.1: D_median within 0.05 of 1 - p = 0.9, and the share of
    market-periods with >= 3 hits within 0.02 of its binomial reference 1 - Binom(50, 0.1).cdf(2).
    Catches: a dispersion index with the wrong normalisation (e.g. Var/Mean^2, or Var of a firm's series),
    hits counted before the burn-in, or a binomial reference with the wrong N."""
    rng = np.random.default_rng(0)
    floor = rng.random((F, T)) < 0.1
    out = burstiness(floor, HOME, BURN_IN)
    assert abs(out['D_median'] - 0.9) < 0.05, out['D_median']
    assert abs(out['D_binomial'] - 0.9) < 0.01, out['D_binomial']
    ref = 1.0 - binom.cdf(2, F // M, 0.1)
    assert abs(out['share_periods_3plus'] - ref) < 0.02, (out['share_periods_3plus'], ref)
    assert abs(out['share_periods_3plus_binom'] - ref) < 0.02, (out['share_periods_3plus_binom'], ref)


def test_market_bad_periods_cluster():
    """Whole-market bad periods with probability 0.05 in which each of the market's firms hits the floor with
    probability 0.6 (no hits otherwise): D_median > 3.
    Catches: a dispersion index computed over firms instead of market-periods (a single firm's 0/1 series
    has Var/Mean = 1 - p < 1 whatever the clustering)."""
    rng = np.random.default_rng(1)
    bad = rng.random((M, T)) < 0.05                       # (M, T) market bad periods
    floor = bad[HOME] & (rng.random((F, T)) < 0.6)        # (F, T)
    out = burstiness(floor, HOME, BURN_IN)
    assert out['D_median'] > 3, out['D_median']
