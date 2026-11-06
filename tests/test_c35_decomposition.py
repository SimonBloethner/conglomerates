"""C35 Task 1: floor / no-floor decomposition of the member - standalone gap (analysis/status_panel.decompose)."""
import numpy as np
import pandas as pd

from analysis.status_panel import decompose, _fe_diff


def _cell(market, decile, member, hit, inc):
    return pd.DataFrame({'market': market, 'decile': decile, 'member': member, 'hit': hit, 'inc': inc})


def _hand_df():
    """Two (market, decile) cells.
    Cell (0, 1): members inc 0.02, 0.02, 0.02, -0.30 (1 of 4 at the floor);
                 standalones 0.01, 0.01, -0.40, -0.40 (2 of 4 at the floor).
    Cell (1, 1): members 0.03, 0.01 (no floor hit); standalones 0.00, 0.02 (no floor hit).
    Cell (1, 2): standalones only (dropped)."""
    return pd.concat([
        _cell(0, 1, [True] * 4, [False, False, False, True], [0.02, 0.02, 0.02, -0.30]),
        _cell(0, 1, [False] * 4, [False, False, True, True], [0.01, 0.01, -0.40, -0.40]),
        _cell(1, 1, [True] * 2, [False, False], [0.03, 0.01]),
        _cell(1, 1, [False] * 2, [False, False], [0.00, 0.02]),
        _cell(1, 2, [False] * 3, [False, True, False], [0.5, -0.5, 0.1]),
    ], ignore_index=True)


def test_hand_decomposition():
    """Exact values by hand.
    Cell (0,1): p_m = 1/4, p_s = 1/2, m_m,nf = 0.02, m_s,nf = 0.01, mean_m = -0.06, mean_s = -0.195, total 0.135;
      A = 0.5 * 0.01 = 0.005; B = -0.30/4 - (-0.80)/4 = 0.125; C = 0.25 * 0.02 = 0.005; A + B + C = 0.135.
    Cell (1,1): p = 0, total = A = 0.02 - 0.01 = 0.01, B = C = 0.
    Weights 4 and 2 (members): total = (4 * 0.135 + 2 * 0.01) / 6, A = (4 * 0.005 + 2 * 0.01) / 6, ...
    Catches: wrong weights, B or C with the wrong sign, the standalone-only cell kept."""
    out = decompose(_hand_df())
    assert out['n_cells'] == 2 and out['n_member_in_cells'] == 6
    assert np.isclose(out['gap_total_mean'], (4 * 0.135 + 2 * 0.01) / 6, atol=1e-14)
    assert np.isclose(out['decomp_ordinary'], (4 * 0.005 + 2 * 0.01) / 6, atol=1e-14)
    assert np.isclose(out['decomp_floor_periods'], 4 * 0.125 / 6, atol=1e-14)
    assert np.isclose(out['decomp_composition'], 4 * 0.005 / 6, atol=1e-14)
    assert abs(out['decomp_residual']) < 1e-15
    assert np.isclose(out['share_ordinary'] + out['share_ruin'], 1.0)
    # no-floor sample: cell (0,1) gap 0.01 (weight 3 members), cell (1,1) gap 0.01 (weight 2)
    assert np.isclose(out['gap_nofloor_mean'], 0.01, atol=1e-14)
    # floor sample: only cell (0,1) has both statuses: -0.30 - (-0.40) = 0.10
    assert np.isclose(out['gap_floor_mean'], 0.10, atol=1e-14)
    # raw floor-sample means over all firm-periods (the standalone-only cell (1,2) included)
    assert np.isclose(out['inc_member_floor_mean'], -0.30)
    assert np.isclose(out['inc_standalone_floor_mean'], (-0.40 - 0.40 - 0.50) / 3)
    assert out['n_nofloor_member'] == 5 and out['n_nofloor_standalone'] == 6


def test_total_equals_panel_gap_on_random_panel():
    """On a random panel in which every cell has both statuses with no-floor periods, the decomposition total
    equals the existing fixed-effects gap (_fe_diff) and A + B + C equals the total to rounding."""
    rng = np.random.default_rng(0)
    n = 20000
    df = pd.DataFrame({'market': rng.integers(0, 5, n), 'decile': rng.integers(1, 11, n),
                       'member': rng.random(n) < 0.4})
    df['hit'] = rng.random(n) < np.where(df.member, 0.03, 0.15)
    df['inc'] = rng.normal(0.0, 0.05, n) + 0.002 * df.member - 0.2 * df.hit
    out = decompose(df)
    fe, _ = _fe_diff(df, 'mean')
    assert out['n_cells'] == 50
    assert np.isclose(out['gap_total_mean'], fe, rtol=0, atol=1e-14)
    assert abs(out['decomp_residual']) < 1e-14
    assert out['share_ruin'] > 0.5          # the floor-rate difference dominates the 0.002 drift by construction
