#!/usr/bin/env python3
"""
C37: pilot D design, the moment rule as the base case -> pilot_d/scenarios.json.

Pilot C (pilot_design.py, pilot_c/, 2,145 scenarios) is the loggain reference and is not changed. Pilot D has the
same blocks, cells, α grids, reps and seeds (cell_seed(cell_id, rep) with pilot C's cell ids), so every pilot D run
of the inherited blocks has a loggain twin in pilot C on the same random numbers. Each pilot D scenario carries:
  pilot        'd'
  twin_id      the pilot C scenario id of its loggain twin (None in the new blocks)
  source_id    the pilot C scenario id of the C36 moment run whose result is reused for it (None otherwise):
               those 255 runs are not rerun, their pickles are copied (analytics/pilot_d.py --copy-c36,
               pilot_d/provenance.csv)
  nu           degrees of freedom of the Student-t family (3.0 for t3, the ν block's value, None otherwise)

Blocks (scenario ids in this order):
- every pilot C block except rule-replay and moment, with decision_rule='moment':
    main 540, equal-split 180, cost-level 360, lookback 100, correlation 45, endogenous-alpha 60,
    floor-level 90, search 120, size 80, cost-level-full 180, cost-level-high 90      = 1,845 (ids 0-1844)
  of which 255 reuse a C36 run (main power law 135, cost-level-high ×5/×20 60, search p = 1 at α 0.1/0.3/0.5 45,
  endogenous α power law 15) and 1,590 are new;
- homogeneous: every market at IQR 0.2 (sigma_range (0.2, 0.2)), 3 families, power law,
  α ∈ {0.05, 0.1, 0.3, 0.5}, 5 reps, under both rules (moment and loggain)        = 120 (ids 1845-1964)
  seeds: the main power-law cell of the family (same draws as the heterogeneous cells up to the IQRs);
- nu: Student-t (log_family 'student_t') with ν ∈ {2.5, 4}, the main IQR range (0.1, 0.3), power law,
  α ∈ {0.1, 0.3, 0.5}, 5 reps, moment rule                                         = 30 (ids 1965-1994)
  seeds: the t3 main power-law cell (same market IQRs as ν = 3).

Total: 1,995 scenarios, 1,740 new runs. The rule-replay block is dropped; its runs stay in pilot C.

C38 appends one block (ids 1995-2009; the 1,995 above are unchanged):
- alpha-0.9: the main power-law cell of each family at a fixed α = 0.9, 5 reps, moment rule = 15, each a copy of
  the main α = 0.5 power-law scenario of its family and rep (same cell id, seed, parameters) with α = 0.9: the
  fixed-rate comparison for the endogenous-α block, whose units adopt α ≈ 0.9. No loggain twin.

C39 appends one block (ids 2010-2054; the 2,010 above are unchanged):
- cv: the main power-law cell of each family at α ∈ {0.1, 0.3, 0.5}, 5 reps, under decision_rule='cv' (the loggain
  replay with its first-order term removed, collaborative_growth.member_gain_cv) = 45, each a copy of the main
  scenario of its family, α and rep (same cell id, seed, parameters): its moment twin is that pilot D main scenario,
  its loggain twin the pilot C scenario in twin_id.

The 27 traced runs (main, power law, 3 families, α ∈ {0.05, 0.1, 0.3}, reps 0-2) are reruns of pilot D main
scenarios with trace_path set (analysis/run_traces.py --pilot d), not separate scenarios.
"""
import json
from pathlib import Path

import pilot_design as pc

DROPPED_BLOCKS = ('rule-replay', 'moment')
HOMOGENEOUS_IQR = 0.2
HOMOGENEOUS_ALPHAS = [0.05, 0.1, 0.3, 0.5]
NU_VALUES = [2.5, 4.0]
NU_ALPHAS = [0.1, 0.3, 0.5]
TOTAL_C37 = 1995
FIXED_ALPHA = 0.9          # C38
TOTAL_C38 = TOTAL_C37 + 3 * pc.N_REPS
CV_ALPHAS = [0.1, 0.3, 0.5]  # C39
TOTAL = TOTAL_C38 + 3 * len(CV_ALPHAS) * pc.N_REPS


def main_power_law_cell(family):
    """cell_id of the main block's (family, power_law) cell in pilot C."""
    return pc.FAMILIES.index(family) * len(pc.COST_FORMS) + pc.COST_FORMS.index('power_law')


def _nu(family):
    return 3.0 if family == 't3' else None


def generate_inherited(c_scenarios, scenario_id=0):
    """Moment twins of every pilot C loggain run outside rule-replay and the C36 moment block."""
    reuse = {s['counterpart_id']: s['scenario_id'] for s in c_scenarios if s['block'] == 'moment'}
    out = []
    for c in c_scenarios:
        if c['block'] in DROPPED_BLOCKS:
            continue
        assert c['decision_rule'] == 'loggain', c['scenario_id']
        s = dict(c)
        s.update({'scenario_id': scenario_id, 'decision_rule': 'moment', 'pilot': 'd',
                  'twin_id': c['scenario_id'], 'source_id': reuse.get(c['scenario_id']),
                  'nu': _nu(c['log_family'])})
        out.append(s)
        scenario_id += 1
    return out, scenario_id


def _power_law_base(family, alpha, rep, cell_id):
    s = pc.make_base_params()
    s.update({
        'cell_id': cell_id, 'log_family': family, 'cost_type': 'power_law',
        'c0': pc.COST_PARAMS['power_law']['c0'], 'c1': pc.COST_PARAMS['power_law']['c1'],
        'c2': pc.COST_PARAMS['power_law']['c2'],
        'lookback': 50, 'cross_corr': 0.0, 'alpha': alpha, 'sharing_rule': pc.SHARING_RULE, 'rep': rep,
        'seed': pc.cell_seed(cell_id, rep), 'alpha_endogenous': False,
        'pilot': 'd', 'twin_id': None, 'source_id': None,
    })
    return s


def generate_homogeneous(scenario_id):
    out = []
    for rule in ('moment', 'loggain'):
        for family in pc.FAMILIES:
            for alpha in HOMOGENEOUS_ALPHAS:
                for rep in range(pc.N_REPS):
                    s = _power_law_base(family, alpha, rep, main_power_law_cell(family))
                    s.update({'block': 'homogeneous', 'scenario_id': scenario_id, 'decision_rule': rule,
                              'sigma_range': [HOMOGENEOUS_IQR, HOMOGENEOUS_IQR], 'nu': _nu(family)})
                    out.append(s)
                    scenario_id += 1
    return out, scenario_id


def generate_nu(scenario_id):
    out = []
    for nu in NU_VALUES:
        for alpha in NU_ALPHAS:
            for rep in range(pc.N_REPS):
                s = _power_law_base('student_t', alpha, rep, main_power_law_cell('t3'))
                s.update({'block': 'nu', 'scenario_id': scenario_id, 'decision_rule': 'moment', 'nu': nu})
                out.append(s)
                scenario_id += 1
    return out, scenario_id


def generate_fixed_alpha(inherited, scenario_id):
    """C38: the main power-law α = 0.5 scenario of each family and rep, at α = FIXED_ALPHA."""
    out = []
    for family in pc.FAMILIES:
        for rep in range(pc.N_REPS):
            src = [s for s in inherited if s['block'] == 'main' and s['cost_type'] == 'power_law'
                   and s['log_family'] == family and s['alpha'] == 0.5 and s['rep'] == rep]
            assert len(src) == 1
            s = dict(src[0])
            s.update({'block': 'alpha-0.9', 'scenario_id': scenario_id, 'alpha': FIXED_ALPHA,
                      'twin_id': None, 'source_id': None})
            out.append(s)
            scenario_id += 1
    return out, scenario_id


def generate_cv(inherited, scenario_id):
    """C39: the main power-law scenario of each family, alpha in CV_ALPHAS and rep, under decision_rule='cv'."""
    out = []
    for family in pc.FAMILIES:
        for alpha in CV_ALPHAS:
            for rep in range(pc.N_REPS):
                src = [s for s in inherited if s['block'] == 'main' and s['cost_type'] == 'power_law'
                       and s['log_family'] == family and s['alpha'] == alpha and s['rep'] == rep]
                assert len(src) == 1
                s = dict(src[0])
                s.update({'block': 'cv', 'scenario_id': scenario_id, 'decision_rule': 'cv', 'source_id': None})
                out.append(s)
                scenario_id += 1
    return out, scenario_id


def generate_all_scenarios():
    c = pc.generate_all_scenarios()
    inherited, sid = generate_inherited(c)
    homog, sid = generate_homogeneous(sid)
    nus, sid = generate_nu(sid)
    assert sid == TOTAL_C37, sid
    fixed, sid = generate_fixed_alpha(inherited, sid)
    assert sid == TOTAL_C38, sid
    cv, sid = generate_cv(inherited, sid)
    scenarios = inherited + homog + nus + fixed + cv
    assert len(scenarios) == TOTAL, len(scenarios)
    return scenarios


def main():
    scenarios = generate_all_scenarios()
    counts = {}
    for s in scenarios:
        counts[s['block']] = counts.get(s['block'], 0) + 1
    print('Pilot D design (moment rule)')
    for b, n in counts.items():
        print(f'  {b}: {n}')
    n_reuse = sum(s['source_id'] is not None for s in scenarios)
    print(f'  Total: {len(scenarios)} ({n_reuse} reuse a C36 run, {len(scenarios) - n_reuse} new)')
    out = Path('pilot_d/scenarios.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(scenarios, f, indent=2)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
