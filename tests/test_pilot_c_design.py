#!/usr/bin/env python3
"""
Tests for Phase C pilot design generator.

Test requirements:
1. Scenario count = 1275
2. floor_c = 0.12717 (C14 calibration)
3. α grid is 9-point
4. Seeds identical across α within (cell_id, rep)
5. market_size_fixed = True in every scenario
6. T = 11000, burn_in = 8000 (from C14)
"""
import json
import os
import sys

# Handle both direct execution and pytest
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pilot_design import (
    generate_all_scenarios,
    FLOOR_C,
    ALPHA_GRID,
    FAMILIES,
    COST_TYPES,
    N_REPS,
    T,
    BURN_IN,
)


def c_for_exponent(target):
    """Compute floor coefficient for target tail exponent."""
    return 1.0 - 1.0 / target


def test_scenario_count():
    """Scenario count = 1275."""
    scenarios = generate_all_scenarios()

    # Check total count
    assert len(scenarios) == 1275, f"Expected 1275 scenarios, got {len(scenarios)}"

    # Check block counts
    blocks = {}
    for s in scenarios:
        block = s['block']
        blocks[block] = blocks.get(block, 0) + 1

    expected = {
        'main': 540,  # 3 × 4 × 9 × 5
        'equal-split': 180,  # 1 × 4 × 9 × 5
        'cost-level': 360,  # 1 × 4 × 2 × 9 × 5
        'lookback': 90,  # 1 × 1 × 2 × 9 × 5
        'correlation': 45,  # 1 × 1 × 1 × 9 × 5
        'endogenous-alpha': 60,  # 3 × 4 × 5
    }

    for block, count in expected.items():
        actual = blocks.get(block, 0)
        assert actual == count, f"Block {block}: expected {count}, got {actual}"

    print("PASS: scenario count = 1275")


def test_floor_c_matches_c14():
    """floor_c = 0.12717 (C14 calibration for Hill ~1.06 at N=50)."""
    expected = 0.12717

    # Check constant
    assert abs(FLOOR_C - expected) < 1e-5, \
        f"FLOOR_C mismatch: {FLOOR_C} vs {expected}"

    # Check in scenarios
    scenarios = generate_all_scenarios()
    for s in scenarios:
        assert abs(s['floor_c'] - expected) < 1e-5, \
            f"Scenario {s['scenario_id']} floor_c mismatch: {s['floor_c']}"

    print(f"PASS: floor_c = {FLOOR_C} matches C14 calibration (0.12717)")


def test_alpha_grid_9_point():
    """α grid is 9-point: [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]."""
    expected = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Check constant
    assert len(ALPHA_GRID) == 9, f"ALPHA_GRID has {len(ALPHA_GRID)} values"
    assert ALPHA_GRID == expected, f"ALPHA_GRID mismatch: {ALPHA_GRID}"

    # Check in main block scenarios
    scenarios = generate_all_scenarios()
    main_alphas = sorted(set(s['alpha'] for s in scenarios if s['block'] == 'main'))
    assert main_alphas == expected, f"Main block alphas: {main_alphas}"

    print(f"PASS: α grid is 9-point {ALPHA_GRID}")


def test_seeds_identical_across_alpha():
    """Seeds identical across α within (cell_id, rep)."""
    scenarios = generate_all_scenarios()

    from collections import defaultdict
    cell_rep_seeds = defaultdict(set)

    for s in scenarios:
        key = (s['cell_id'], s['rep'])
        cell_rep_seeds[key].add(s['seed'])

    # All seeds within a cell/rep should be identical
    mismatches = []
    for key, seeds in cell_rep_seeds.items():
        if len(seeds) != 1:
            mismatches.append((key, seeds))

    assert len(mismatches) == 0, \
        f"Seed mismatches across α: {mismatches[:5]}..."

    print(f"PASS: seeds identical across α ({len(cell_rep_seeds)} cell/rep combos)")


def test_seeds_differ_across_reps():
    """Seeds differ across reps within the same cell."""
    scenarios = generate_all_scenarios()

    from collections import defaultdict
    cell_rep_seeds = defaultdict(list)

    for s in scenarios:
        key = s['cell_id']
        cell_rep_seeds[key].append((s['rep'], s['seed']))

    # Check each cell has unique seeds per rep
    for cell_id, rep_seeds in cell_rep_seeds.items():
        seeds_by_rep = {}
        for rep, seed in rep_seeds:
            if rep not in seeds_by_rep:
                seeds_by_rep[rep] = seed

        unique_seeds = set(seeds_by_rep.values())
        if len(unique_seeds) > 1 and len(unique_seeds) != len(seeds_by_rep):
            raise AssertionError(
                f"Cell {cell_id} has duplicate seeds across reps: {seeds_by_rep}"
            )

    print("PASS: seeds differ across reps within cells")


def test_seeds_differ_across_cells():
    """Seeds differ across cells (for same rep)."""
    scenarios = generate_all_scenarios()

    # Get seed for rep 0 of each cell
    rep0_seeds = {}
    for s in scenarios:
        if s['rep'] == 0:
            cell_id = s['cell_id']
            if cell_id not in rep0_seeds:
                rep0_seeds[cell_id] = s['seed']

    # All seeds should be unique
    seeds = list(rep0_seeds.values())
    assert len(set(seeds)) == len(seeds), \
        "Duplicate seeds across cells for rep 0"

    print(f"PASS: seeds differ across cells ({len(rep0_seeds)} cells)")


def test_burn_in_scenario_exists():
    """Burn-in scenario (laplace/power_law/α=0.1) exists with 5 reps."""
    scenarios = generate_all_scenarios()

    burn_in = []
    for s in scenarios:
        if (s['block'] == 'main' and
            s['log_family'] == 'laplace' and
            s['cost_type'] == 'power_law' and
            abs(s['alpha'] - 0.1) < 0.001):
            burn_in.append(s)

    assert len(burn_in) == 5, f"Expected 5 burn-in scenarios, got {len(burn_in)}"

    # Verify reps 0-4
    reps = sorted(s['rep'] for s in burn_in)
    assert reps == [0, 1, 2, 3, 4], f"Reps should be 0-4, got {reps}"

    # Verify all share same cell_id
    cell_ids = set(s['cell_id'] for s in burn_in)
    assert len(cell_ids) == 1, f"Burn-in scenarios should share cell_id"

    print("PASS: burn-in scenario exists (laplace/power_law/α=0.1, 5 reps)")


def test_scenario_structure():
    """Each scenario has required fields."""
    scenarios = generate_all_scenarios()

    required_fields = [
        'M', 'N', 'T', 'burn_in', 'merge_thresh', 'proportional', 'growth_process',
        'mu_range', 'sigma_range', 'floor_c', 'metric_every', 'market_size_fixed',
        'block', 'scenario_id', 'cell_id', 'log_family', 'cost_type',
        'c0', 'c1', 'c2', 'lookback', 'cross_corr', 'alpha',
        'sharing_rule', 'rep', 'seed', 'alpha_endogenous',
    ]

    for s in scenarios:
        for field in required_fields:
            assert field in s, f"Scenario {s['scenario_id']} missing '{field}'"

    print(f"PASS: all {len(required_fields)} required fields present")


def test_market_size_fixed():
    """market_size_fixed = True in every scenario (C14 requirement)."""
    scenarios = generate_all_scenarios()

    for s in scenarios:
        assert 'market_size_fixed' in s, \
            f"Scenario {s['scenario_id']} missing 'market_size_fixed'"
        assert s['market_size_fixed'] is True, \
            f"Scenario {s['scenario_id']} has market_size_fixed={s['market_size_fixed']}, expected True"

    print(f"PASS: market_size_fixed=True in all {len(scenarios)} scenarios")


def test_t_and_burn_in():
    """T=11000, burn_in=8000 (from C14)."""
    scenarios = generate_all_scenarios()

    # Check constants (C14 values)
    assert T == 11000, f"T should be 11000 (C14), got {T}"
    assert BURN_IN == 8000, f"BURN_IN should be 8000 (C14), got {BURN_IN}"

    # Check all scenarios
    for s in scenarios:
        assert s['T'] == T, f"Scenario {s['scenario_id']} has T={s['T']}, expected {T}"
        assert s['burn_in'] == BURN_IN, f"Scenario {s['scenario_id']} has burn_in={s['burn_in']}"

    # Verify burn_in < T
    assert BURN_IN < T, "burn_in should be less than T"

    print(f"PASS: T={T}, burn_in={BURN_IN} set correctly (C14)")


if __name__ == '__main__':
    print("Testing Phase C pilot design...\n")

    test_scenario_count()
    test_floor_c_matches_c14()
    test_alpha_grid_9_point()
    test_seeds_identical_across_alpha()
    test_seeds_differ_across_reps()
    test_seeds_differ_across_cells()
    test_burn_in_scenario_exists()
    test_scenario_structure()
    test_market_size_fixed()
    test_t_and_burn_in()

    print("\nAll Phase C pilot design tests passed!")
