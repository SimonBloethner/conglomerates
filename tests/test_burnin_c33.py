#!/usr/bin/env python3
"""
Tests for the C33 burn-in diagnostic (run_burnin_c33.py).

1. The 20 diagnostic scenarios are the main-block pilot scenarios with the pilot's seeds
2. block_means averages per-step series over metric_every blocks
3. A series that settles early converges before burn-in
4. A drifting series is reported as never converged, not as converged at step T
5. A series that is identically zero or absent is not applicable
6. extract_series reproduces the model's own post-burn-in summary
7. Extension: window difference and trend are recovered from known series
8. Extension: the decision rule fixed before the T = 16000 runs
"""
import os
import sys

import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pilot_design import BURN_IN, T, cell_seed
from run_burnin_c33 import analyse_series, block_means, burnin_scenarios, extract_series, run_scenario
from run_burnin_c33_ext import keeps_burn_in, trend_after_burn_in, window_comparison

METRIC_EVERY = 100
N_OBS = T // METRIC_EVERY


def test_scenarios_are_the_pilot_runs():
    """20 runs: laplace at α 0, 0.1, 0.3 (cell 7) and t3 at α 0.3 (cell 11), reps 0-4, pilot seeds."""
    scenarios = burnin_scenarios()
    assert len(scenarios) == 20

    keys = [(s['log_family'], s['alpha'], s['rep']) for s in scenarios]
    expected = [(family, alpha, rep)
                for family, alpha in [('laplace', 0.0), ('laplace', 0.1), ('laplace', 0.3), ('t3', 0.3)]
                for rep in range(5)]
    assert keys == expected

    for s in scenarios:
        assert s['block'] == 'main' and s['cost_type'] == 'power_law'
        assert s['cell_id'] == (7 if s['log_family'] == 'laplace' else 11)
        assert s['seed'] == cell_seed(s['cell_id'], s['rep'])
        assert s['T'] == 11000 and s['burn_in'] == 8000
        assert s['g'] == 0.055 and s['lookback'] == 50


def test_block_means():
    per_step = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    assert np.array_equal(block_means(per_step, 2), [2.0, 6.0, 10.0])


def test_settled_series_converges_before_burn_in():
    """Transient until step 2000, then constant: converges and passes Geweke."""
    rng = np.random.default_rng(0)
    series = np.full(N_OBS, 1.0) + 1e-4 * rng.standard_normal(N_OBS)
    series[:15] = 5.0
    verdict = analyse_series(np.tile(series, (5, 1)), METRIC_EVERY)

    assert verdict['applicable']
    assert verdict['converged']
    assert 1500 < verdict['conv_step'] <= 2500
    assert verdict['max_dev'] < 0.02
    assert verdict['geweke_pass']


def test_drifting_series_is_not_reported_as_converged():
    """A sustained drift never settles: conv_step is None, not T."""
    series = 1.0 + np.linspace(0.0, 1.0, N_OBS)
    verdict = analyse_series(np.tile(series, (5, 1)), METRIC_EVERY)

    assert verdict['conv_step'] is None
    assert not verdict['converged']
    assert verdict['max_dev'] > 0.02
    assert not verdict['geweke_pass']


def test_absent_series_are_not_applicable():
    assert not analyse_series(np.zeros((5, N_OBS)), METRIC_EVERY)['applicable']
    assert not analyse_series(np.full((5, N_OBS), np.nan), METRIC_EVERY)['applicable']


def test_extract_series_matches_model_summary():
    """Series means over t >= burn_in equal the model's summary scalars (short run)."""
    scenario = dict(burnin_scenarios()[10], T=600, burn_in=300, M=10, N=20)
    result = run_scenario(scenario)
    series = extract_series(result, METRIC_EVERY)
    summary = result[-1]['summary']

    for values in series.values():
        assert values.shape == (6,)

    assert np.mean(series['fhr_standalone'][3:]) == np.float64(summary['floor_hit_rate_standalone'])
    assert np.mean(series['fhr_member'][3:]) == np.float64(summary['floor_hit_rate_member'])
    assert np.isclose(np.mean(series['mergers'][3:]), summary['mergers_per_period'])
    assert np.isclose(np.mean(series['floor_exits'][3:]), summary['floor_exits_per_period'])
    assert np.isclose(np.mean(series['voluntary_exits'][3:]),
                      summary['exits_per_period'] - summary['floor_exits_per_period'])


def test_window_comparison_and_trend():
    """A 1% step at T shows up as a +1% window difference; a linear trend is recovered in %/1000 steps."""
    steps = np.arange(1, 161) * METRIC_EVERY
    runs = np.tile(np.where(steps > T, 1.01, 1.0), (5, 1)) + 1e-6 * np.arange(5)[:, None]
    window = window_comparison(runs, steps)
    assert abs(window['diff'] - 1.0) < 1e-3
    assert window['ci_low'] <= window['diff'] <= window['ci_high']

    ramp = 1.0 + 0.001 * (steps - 12000) / 1000
    trend = trend_after_burn_in(np.tile(ramp, (5, 1)) + 1e-6 * np.arange(5)[:, None], steps)
    assert abs(trend['slope'] - 0.1) < 1e-3
    assert abs(trend['implied_change'] - 0.8) < 1e-2


def test_decision_rule():
    """burn_in stands only if every window difference and the t3 capital-share trend are inside 2%."""
    small = {'diff': 1.9}
    flat = {'implied_change': -1.9}
    assert keeps_burn_in([small, {'diff': -1.5}], flat)
    assert not keeps_burn_in([small, {'diff': -2.1}], flat)
    assert not keeps_burn_in([small], {'implied_change': -2.0})
