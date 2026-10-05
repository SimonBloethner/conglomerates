#!/usr/bin/env python3
"""
Phase C pilot design generator.

Generates pilot_c/scenarios.json with factorial design for Phase C experiments.

Fixed parameters:
- M=N=50, merge_thresh=0.05, proportional=False
- growth_process=log_family, mu_range=(0.01, 0.1), sigma_range=(0.1, 0.3)
- sharing_rule=proportional (default), floor_c from C1 benchmark
- metric_every=100, 5 reps, common random numbers
- α grid: [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

Blocks:
- main: 3 families × 4 costs × 9 α × 5 reps = 540 runs
- equal-split: normal × 4 costs × 9 α × 5 reps = 180 runs
- cost level: laplace × 4 costs × 2 multipliers × 9 α × 5 reps = 360 runs
- lookback: laplace × power_law × 2 lookbacks × 9 α × 5 reps = 90 runs
- correlation: laplace × power_law × cross_corr=0.3 × 9 α × 5 reps = 45 runs
- endogenous α: 3 families × 4 costs × 5 reps = 60 runs (α_start=0.1)

Total: 1275 runs
"""
import json
import os
from pathlib import Path


# ============================================================================
# FIXED PARAMETERS
# ============================================================================

M = 50
N = 50
MERGE_THRESH = 0.05
PROPORTIONAL = False  # Management cost NOT proportional to size
GROWTH_PROCESS = 'log_family'
MU_RANGE = (0.01, 0.1)
SIGMA_RANGE = (0.1, 0.3)  # IQR range
SHARING_RULE = 'proportional'
METRIC_EVERY = 100
N_REPS = 5

# Floor coefficient from C1: c_for_exponent(1.06) = 1 - 1/1.06
FLOOR_C = 1.0 - 1.0 / 1.06  # ≈ 0.0566037736

# Alpha grid (9 values)
ALPHA_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# Distribution families
FAMILIES = ['normal', 'laplace', 't3']

# Cost types with calibrated defaults (from benchmarks.py)
COST_TYPES = ['linear', 'quadratic', 'exponential', 'power_law']

# Cost parameters (calibrated for convergence at size=40)
COST_PARAMS = {
    'power_law': {'c0': 0.00002032, 'c1': 1.2, 'c2': 0.001},
    'linear': {'c0': 0.00001, 'c1': 0.00004225, 'c2': 0.001},
    'quadratic': {'c0': 0.0001, 'c1': 0.000001, 'c2': 0.00000097},
    'exponential': {'c0': 0.001, 'c1': 0.01326571, 'c2': 0.001},
}

# Base seed for common random numbers
BASE_SEED = 42

# Burn-in parameters (from burn_in_analysis.py)
T = 6000
BURN_IN = 2000


# ============================================================================
# SCENARIO GENERATION
# ============================================================================

def make_base_params():
    """Return fixed parameters common to all scenarios."""
    return {
        'M': M,
        'N': N,
        'T': T,
        'burn_in': BURN_IN,
        'merge_thresh': MERGE_THRESH,
        'proportional': PROPORTIONAL,
        'growth_process': GROWTH_PROCESS,
        'mu_range': list(MU_RANGE),
        'sigma_range': list(SIGMA_RANGE),
        'floor_c': FLOOR_C,
        'metric_every': METRIC_EVERY,
    }


def cell_seed(cell_id, rep):
    """
    Compute seed for (cell, rep).

    Seeds are identical across α within a cell and differ across reps.
    A cell is defined by (block, family, cost_type, lookback, cross_corr, sharing_rule).
    This ensures common random numbers for comparing different α values.
    """
    return BASE_SEED + cell_id * 100 + rep


def generate_main_block(scenarios, scenario_id):
    """
    Main block: 3 families × 4 costs × 9 α × 5 reps = 540 runs
    """
    cell_id = 0
    for family in FAMILIES:
        for cost_type in COST_TYPES:
            # All α values within this (family, cost_type) share seeds per rep
            for alpha in ALPHA_GRID:
                for rep in range(N_REPS):
                    s = make_base_params()
                    s.update({
                        'block': 'main',
                        'scenario_id': scenario_id,
                        'cell_id': cell_id,
                        'log_family': family,
                        'cost_type': cost_type,
                        'c0': COST_PARAMS[cost_type]['c0'],
                        'c1': COST_PARAMS[cost_type]['c1'],
                        'c2': COST_PARAMS[cost_type]['c2'],
                        'lookback': 50,
                        'cross_corr': 0.0,
                        'alpha': alpha,
                        'sharing_rule': SHARING_RULE,
                        'rep': rep,
                        'seed': cell_seed(cell_id, rep),
                        'alpha_endogenous': False,
                    })
                    scenarios.append(s)
                    scenario_id += 1
            cell_id += 1
    return scenario_id, cell_id


def generate_equal_split_block(scenarios, scenario_id, cell_id):
    """
    Equal-split check: normal × 4 costs × 9 α × 5 reps = 180 runs
    sharing_rule=equal
    """
    family = 'normal'
    for cost_type in COST_TYPES:
        # All α values within this (cost_type) share seeds per rep
        for alpha in ALPHA_GRID:
            for rep in range(N_REPS):
                s = make_base_params()
                s.update({
                    'block': 'equal-split',
                    'scenario_id': scenario_id,
                    'cell_id': cell_id,
                    'log_family': family,
                    'cost_type': cost_type,
                    'c0': COST_PARAMS[cost_type]['c0'],
                    'c1': COST_PARAMS[cost_type]['c1'],
                    'c2': COST_PARAMS[cost_type]['c2'],
                    'lookback': 50,
                    'cross_corr': 0.0,
                    'alpha': alpha,
                    'sharing_rule': 'equal',
                    'rep': rep,
                    'seed': cell_seed(cell_id, rep),
                    'alpha_endogenous': False,
                })
                scenarios.append(s)
                scenario_id += 1
        cell_id += 1
    return scenario_id, cell_id


def generate_cost_level_block(scenarios, scenario_id, cell_id):
    """
    Cost level: laplace × 4 costs × 2 multipliers × 9 α × 5 reps = 360 runs
    Multipliers: 0.5 and 2.0 applied to c0
    """
    family = 'laplace'
    multipliers = [0.5, 2.0]

    for cost_type in COST_TYPES:
        for mult in multipliers:
            # All α values within this (cost_type, mult) share seeds per rep
            for alpha in ALPHA_GRID:
                for rep in range(N_REPS):
                    s = make_base_params()
                    s.update({
                        'block': 'cost-level',
                        'scenario_id': scenario_id,
                        'cell_id': cell_id,
                        'log_family': family,
                        'cost_type': cost_type,
                        'c0': COST_PARAMS[cost_type]['c0'] * mult,
                        'c1': COST_PARAMS[cost_type]['c1'],
                        'c2': COST_PARAMS[cost_type]['c2'],
                        'cost_multiplier': mult,
                        'lookback': 50,
                        'cross_corr': 0.0,
                        'alpha': alpha,
                        'sharing_rule': SHARING_RULE,
                        'rep': rep,
                        'seed': cell_seed(cell_id, rep),
                        'alpha_endogenous': False,
                    })
                    scenarios.append(s)
                    scenario_id += 1
            cell_id += 1
    return scenario_id, cell_id


def generate_lookback_block(scenarios, scenario_id, cell_id):
    """
    Lookback: laplace × power_law × 2 lookbacks × 9 α × 5 reps = 90 runs
    Lookbacks: 200, 1000
    """
    family = 'laplace'
    cost_type = 'power_law'
    lookbacks = [200, 1000]

    for lookback in lookbacks:
        # All α values within this (lookback) share seeds per rep
        for alpha in ALPHA_GRID:
            for rep in range(N_REPS):
                s = make_base_params()
                s.update({
                    'block': 'lookback',
                    'scenario_id': scenario_id,
                    'cell_id': cell_id,
                    'log_family': family,
                    'cost_type': cost_type,
                    'c0': COST_PARAMS[cost_type]['c0'],
                    'c1': COST_PARAMS[cost_type]['c1'],
                    'c2': COST_PARAMS[cost_type]['c2'],
                    'lookback': lookback,
                    'cross_corr': 0.0,
                    'alpha': alpha,
                    'sharing_rule': SHARING_RULE,
                    'rep': rep,
                    'seed': cell_seed(cell_id, rep),
                    'alpha_endogenous': False,
                })
                scenarios.append(s)
                scenario_id += 1
        cell_id += 1
    return scenario_id, cell_id


def generate_correlation_block(scenarios, scenario_id, cell_id):
    """
    Correlation: laplace × power_law × cross_corr=0.3 × 9 α × 5 reps = 45 runs
    """
    family = 'laplace'
    cost_type = 'power_law'

    # All α values share seeds per rep (one cell for this block)
    for alpha in ALPHA_GRID:
        for rep in range(N_REPS):
            s = make_base_params()
            s.update({
                'block': 'correlation',
                'scenario_id': scenario_id,
                'cell_id': cell_id,
                'log_family': family,
                'cost_type': cost_type,
                'c0': COST_PARAMS[cost_type]['c0'],
                'c1': COST_PARAMS[cost_type]['c1'],
                'c2': COST_PARAMS[cost_type]['c2'],
                'lookback': 50,
                'cross_corr': 0.3,
                'alpha': alpha,
                'sharing_rule': SHARING_RULE,
                'rep': rep,
                'seed': cell_seed(cell_id, rep),
                'alpha_endogenous': False,
            })
            scenarios.append(s)
            scenario_id += 1
    cell_id += 1
    return scenario_id, cell_id


def generate_endogenous_alpha_block(scenarios, scenario_id, cell_id):
    """
    Endogenous α: 3 families × 4 costs × 5 reps = 60 runs
    alpha_endogenous=True, start α=0.1
    """
    for family in FAMILIES:
        for cost_type in COST_TYPES:
            # Each (family, cost_type) is a cell
            for rep in range(N_REPS):
                s = make_base_params()
                s.update({
                    'block': 'endogenous-alpha',
                    'scenario_id': scenario_id,
                    'cell_id': cell_id,
                    'log_family': family,
                    'cost_type': cost_type,
                    'c0': COST_PARAMS[cost_type]['c0'],
                    'c1': COST_PARAMS[cost_type]['c1'],
                    'c2': COST_PARAMS[cost_type]['c2'],
                    'lookback': 50,
                    'cross_corr': 0.0,
                    'alpha': 0.1,  # Start alpha
                    'sharing_rule': SHARING_RULE,
                    'rep': rep,
                    'seed': cell_seed(cell_id, rep),
                    'alpha_endogenous': True,
                })
                scenarios.append(s)
                scenario_id += 1
            cell_id += 1
    return scenario_id, cell_id


def generate_all_scenarios():
    """Generate all scenarios for Phase C pilot."""
    scenarios = []
    scenario_id = 0
    cell_id = 0

    scenario_id, cell_id = generate_main_block(scenarios, scenario_id)
    scenario_id, cell_id = generate_equal_split_block(scenarios, scenario_id, cell_id)
    scenario_id, cell_id = generate_cost_level_block(scenarios, scenario_id, cell_id)
    scenario_id, cell_id = generate_lookback_block(scenarios, scenario_id, cell_id)
    scenario_id, cell_id = generate_correlation_block(scenarios, scenario_id, cell_id)
    scenario_id, cell_id = generate_endogenous_alpha_block(scenarios, scenario_id, cell_id)

    return scenarios


def set_burn_in_params(scenarios, T, burn_in):
    """Set T and burn_in for all scenarios."""
    for s in scenarios:
        s['T'] = T
        s['burn_in'] = burn_in


def write_scenarios_json(scenarios, output_path='pilot_c/scenarios.json'):
    """Write scenarios to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(scenarios, f, indent=2)

    print(f"Wrote {len(scenarios)} scenarios to {output_path}")
    return output_path


def get_burn_in_scenario(scenarios):
    """
    Get the burn-in scenario: main block, laplace, power_law, α=0.1.
    Returns one scenario per rep (5 total).
    """
    burn_in_scenarios = []
    for s in scenarios:
        if (s['block'] == 'main' and
            s['log_family'] == 'laplace' and
            s['cost_type'] == 'power_law' and
            abs(s['alpha'] - 0.1) < 0.001):
            burn_in_scenarios.append(s)
    return burn_in_scenarios


def main():
    """Generate Phase C pilot design."""
    scenarios = generate_all_scenarios()

    # Count by block
    blocks = {}
    for s in scenarios:
        block = s['block']
        blocks[block] = blocks.get(block, 0) + 1

    print("Phase C Pilot Design")
    print("=" * 50)
    print(f"Fixed parameters:")
    print(f"  M=N={M}, T={T}, burn_in={BURN_IN}")
    print(f"  merge_thresh={MERGE_THRESH}")
    print(f"  growth_process={GROWTH_PROCESS}")
    print(f"  mu_range={MU_RANGE}, sigma_range={SIGMA_RANGE}")
    print(f"  floor_c={FLOOR_C:.10f} (exponent=1.06)")
    print(f"  metric_every={METRIC_EVERY}")
    print(f"  α grid: {ALPHA_GRID}")
    print()
    print("Blocks:")
    for block, count in blocks.items():
        print(f"  {block}: {count} runs")
    print(f"  Total: {len(scenarios)} runs")
    print()

    # Note: T and burn_in will be set after burn-in analysis
    # For now, write scenarios without T/burn_in
    output_path = write_scenarios_json(scenarios)

    # Print burn-in scenario info
    burn_in = get_burn_in_scenario(scenarios)
    print(f"\nBurn-in scenarios (laplace/power_law/α=0.1): {len(burn_in)} runs")

    return scenarios


if __name__ == '__main__':
    main()
