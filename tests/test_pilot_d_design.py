"""C37: pilot D design (pilot_design_d.py, pilot_d/scenarios.json), the moment rule as the base case.

1,995 scenarios: the 1,845 moment twins of pilot C's loggain runs outside rule-replay and the C36 moment block
(255 of them reuse a C36 run), the homogeneous block (120, both rules) and the nu block (30).
C38 appends the alpha-0.9 block (15, ids 1995-2009): 2,010 scenarios, the first 1,995 unchanged.
C39 appends the cv block (45, ids 2010-2054): 2,055 scenarios, the first 2,010 unchanged."""
import gzip
import json
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
sys.path.insert(0, os.path.join(ROOT, 'analytics'))

import pilot_design as pc  # noqa: E402
import pilot_design_d as pdd  # noqa: E402
from pilot_d import model_keys_match  # noqa: E402
from pilots import pop_pilot, pilot_dir  # noqa: E402

TOTAL_C37 = 1995
BLOCK_COUNTS_C37 = {'main': 540, 'equal-split': 180, 'cost-level': 360, 'lookback': 100, 'correlation': 45,
                    'endogenous-alpha': 60, 'floor-level': 90, 'search': 120, 'size': 80, 'cost-level-full': 180,
                    'cost-level-high': 90, 'homogeneous': 120, 'nu': 30}
BLOCK_COUNTS = dict(BLOCK_COUNTS_C37, **{'alpha-0.9': 15, 'cv': 45})       # C38, C39
TOTAL_C38 = TOTAL_C37 + 15
TOTAL = TOTAL_C38 + 45
NEW_BLOCKS = ('homogeneous', 'nu', 'alpha-0.9', 'cv')             # not inherited from pilot C
D_KEYS = {'pilot', 'twin_id', 'source_id', 'nu'}


@pytest.fixture(scope='module')
def d():
    return pdd.generate_all_scenarios()


@pytest.fixture(scope='module')
def c():
    return {s['scenario_id']: s for s in pc.generate_all_scenarios()}


def test_counts(d):
    assert len(d) == TOTAL == sum(BLOCK_COUNTS.values())
    counts = {}
    for s in d:
        counts[s['block']] = counts.get(s['block'], 0) + 1
    assert counts == BLOCK_COUNTS
    assert [s['scenario_id'] for s in d] == list(range(TOTAL))
    assert sum(s['source_id'] is not None for s in d) == 255
    assert TOTAL_C37 - 255 == 1740                                # new runs in C37 (C38 adds 15)
    assert all(s['pilot'] == 'd' for s in d)


def test_json_matches_generator(d):
    with open(os.path.join(ROOT, 'pilot_d', 'scenarios.json')) as f:
        assert json.load(f) == d


def test_c38_alpha_09_block(d):
    """C38: 15 copies of the main power-law alpha = 0.5 scenarios (same seed, cell, parameters) at alpha = 0.9,
    appended after the C37 scenarios, which are unchanged."""
    f = d[TOTAL_C37:TOTAL_C38]
    assert [s['block'] for s in f] == ['alpha-0.9'] * 15 and [s['scenario_id'] for s in f] == list(range(1995, 2010))
    for s in f:
        src = [t for t in d[:TOTAL_C37] if t['block'] == 'main' and t['cost_type'] == 'power_law'
               and t['log_family'] == s['log_family'] and t['alpha'] == 0.5 and t['rep'] == s['rep']]
        assert len(src) == 1
        t = src[0]
        assert s['alpha'] == 0.9 and s['decision_rule'] == 'moment' and s['twin_id'] is None and s['source_id'] is None
        assert s['seed'] == pc.cell_seed(t['cell_id'], s['rep']) == t['seed']
        assert set(s) == set(t)
        assert all(s[k] == t[k] for k in t if k not in ('scenario_id', 'block', 'alpha', 'twin_id', 'source_id'))


def test_inherited_runs_are_moment_twins_of_pilot_c(d, c):
    """Every inherited scenario equals its pilot C loggain twin except decision_rule, the id and the pilot D keys:
    same cell id, seed, rep and parameters, so both rules run on the same random numbers."""
    inherited = [s for s in d if s['block'] not in NEW_BLOCKS]
    assert len(inherited) == 1845
    twins = [s['twin_id'] for s in inherited]
    assert len(set(twins)) == len(twins)
    for s in inherited:
        t = c[s['twin_id']]
        assert t['decision_rule'] == 'loggain' and s['decision_rule'] == 'moment'
        assert t['block'] not in ('rule-replay', 'moment')
        assert set(s) - D_KEYS == set(t), s['scenario_id']
        assert all(s[k] == t[k] for k in t if k not in ('scenario_id', 'decision_rule')), s['scenario_id']
        assert s['nu'] == (3.0 if s['log_family'] == 't3' else None)
    # every loggain run of pilot C outside the two dropped blocks has exactly one twin
    assert sorted(twins) == sorted(i for i, t in c.items() if t['block'] not in ('rule-replay', 'moment'))


def test_reused_c36_runs(d, c):
    """The 255 reused runs are exactly the C36 moment block, each the same model run as its pilot D scenario."""
    reused = [s for s in d if s['source_id'] is not None]
    assert sorted(s['source_id'] for s in reused) == sorted(i for i, t in c.items() if t['block'] == 'moment')
    for s in reused:
        assert model_keys_match(c[s['source_id']], s), s['scenario_id']
    by = {}
    for s in reused:
        by[s['block']] = by.get(s['block'], 0) + 1
    assert by == {'main': 135, 'cost-level-high': 60, 'search': 45, 'endogenous-alpha': 15}


def test_model_keys_match_rejects_a_changed_parameter(d, c):
    s = next(x for x in d if x['source_id'] is not None)
    bad = dict(s, alpha=s['alpha'] + 0.1)
    assert not model_keys_match(c[s['source_id']], bad)
    assert not model_keys_match(c[s['source_id']], dict(s, nu=2.5))


def test_no_replay_and_rule_by_block(d):
    assert not any(s['decision_rule'] == 'replay' for s in d)
    for s in d:
        if s['block'] == 'cv':
            assert s['decision_rule'] == 'cv'
        elif s['block'] != 'homogeneous':
            assert s['decision_rule'] == 'moment'
    assert sum(s['decision_rule'] == 'loggain' for s in d) == 60


def test_homogeneous_block(d, c):
    h = [s for s in d if s['block'] == 'homogeneous']
    assert len(h) == 120
    assert all(s['sigma_range'] == [0.2, 0.2] and s['cost_type'] == 'power_law' and s['twin_id'] is None for s in h)
    assert sorted({s['alpha'] for s in h}) == [0.05, 0.1, 0.3, 0.5]
    keys = {}
    for s in h:
        main = [t for t in c.values() if t['block'] == 'main' and t['cost_type'] == 'power_law'
                and t['log_family'] == s['log_family'] and t['alpha'] == s['alpha'] and t['rep'] == s['rep']]
        assert len(main) == 1 and s['seed'] == main[0]['seed'] and s['cell_id'] == main[0]['cell_id']
        keys.setdefault((s['log_family'], s['alpha'], s['rep']), set()).add(s['decision_rule'])
        # identical to the main power-law cell except the IQR range, the rule and the block
        assert all(s[k] == main[0][k] for k in main[0]
                   if k not in ('scenario_id', 'block', 'sigma_range', 'decision_rule'))
    assert len(keys) == 60 and all(v == {'moment', 'loggain'} for v in keys.values())


def test_nu_block(d, c):
    n = [s for s in d if s['block'] == 'nu']
    assert len(n) == 30
    assert sorted({s['nu'] for s in n}) == [2.5, 4.0]
    assert sorted({s['alpha'] for s in n}) == [0.1, 0.3, 0.5]
    t3 = {(t['alpha'], t['rep']): t for t in c.values()
          if t['block'] == 'main' and t['cost_type'] == 'power_law' and t['log_family'] == 't3'}
    for s in n:
        assert s['log_family'] == 'student_t' and s['sigma_range'] == [0.1, 0.3]
        assert s['seed'] == t3[(s['alpha'], s['rep'])]['seed']


def test_traced_selection(d):
    """analysis/run_traces.select picks the 27 traced cells from pilot D, all under the moment rule."""
    from run_traces import select
    chosen = select(d)
    assert len(chosen) == 27 and all(s['decision_rule'] == 'moment' and s['source_id'] is not None for s in chosen)


def test_pilot_c_design_unchanged():
    with open(os.path.join(ROOT, 'pilot_c', 'scenarios.json')) as f:
        assert json.load(f) == pc.generate_all_scenarios()


def test_pop_pilot():
    argv = ['x', '3', '--pilot', 'd', '--out', 'y']
    assert pop_pilot(argv) == 'd' and argv == ['x', '3', '--out', 'y']
    assert pop_pilot(['x']) == 'c'
    with pytest.raises(SystemExit):
        pop_pilot(['x', '--pilot', 'e'])
    assert pilot_dir('d', 'results') == os.path.join('pilot_d', 'results')


def test_runner_passes_nu():
    import run_pilot_c
    src = open(run_pilot_c.__file__).read()
    assert "nu=scenario.get('nu') or 3.0" in src


def test_c39_cv_block(d, c):
    """C39: 45 copies of the main power-law scenarios at alpha 0.1, 0.3, 0.5 (same seed, cell, parameters, loggain
    twin) under decision_rule='cv', appended after the C38 scenarios, which are unchanged."""
    f = d[TOTAL_C38:]
    assert [s['block'] for s in f] == ['cv'] * 45 and [s['scenario_id'] for s in f] == list(range(2010, 2055))
    assert sorted({s['alpha'] for s in f}) == [0.1, 0.3, 0.5]
    for s in f:
        src = [t for t in d[:TOTAL_C37] if t['block'] == 'main' and t['cost_type'] == 'power_law'
               and t['log_family'] == s['log_family'] and t['alpha'] == s['alpha'] and t['rep'] == s['rep']]
        assert len(src) == 1
        t = src[0]
        assert s['decision_rule'] == 'cv' and s['source_id'] is None and s['twin_id'] == t['twin_id']
        assert c[s['twin_id']]['decision_rule'] == 'loggain' and s['seed'] == t['seed'] == c[s['twin_id']]['seed']
        assert set(s) == set(t)
        assert all(s[k] == t[k] for k in t if k not in ('scenario_id', 'block', 'decision_rule', 'source_id'))
