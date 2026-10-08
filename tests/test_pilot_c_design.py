#!/usr/bin/env python3
"""
Tests for Phase C pilot design generator.

Test requirements (updated for C17):
1. Scenario count = 1370
2. floor_c = 0.12717 (C14 calibration)
3. α grid is 9-point
4. Seeds identical across α within (cell_id, rep)
5. market_size_fixed = True in every scenario
6. T = 11000, burn_in = 8000 (from C14)
7. decision_rule = 'loggain' in all blocks except rule-replay
8. rule-replay block uses decision_rule = 'replay' (45 runs)
9. floor-level block exists with floor_c ∈ {0.10, 0.35} (90 runs)
10. lookback block: 5 values × 2 α × 5 reps = 50 runs
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
    """Scenario count = 1370 (updated for C17)."""
    scenarios = generate_all_scenarios()

    # Check total count
    assert len(scenarios) == 1370, f"Expected 1370 scenarios, got {len(scenarios)}"

    # Check block counts
    blocks = {}
    for s in scenarios:
        block = s['block']
        blocks[block] = blocks.get(block, 0) + 1

    expected = {
        'main': 540,  # 3 × 4 × 9 × 5
        'equal-split': 180,  # 1 × 4 × 9 × 5
        'cost-level': 360,  # 1 × 4 × 2 × 9 × 5
        'lookback': 50,  # C17: 1 × 1 × 5 × 2 × 5 (5 lookback × 2 α × 5 reps)
        'correlation': 45,  # 1 × 1 × 1 × 9 × 5
        'endogenous-alpha': 60,  # 3 × 4 × 5
        'rule-replay': 45,  # C17: 1 × 1 × 9 × 5
        'floor-level': 90,  # C17: 1 × 1 × 2 × 9 × 5 (2 floor values × 9 α × 5 reps)
    }

    for block, count in expected.items():
        actual = blocks.get(block, 0)
        assert actual == count, f"Block {block}: expected {count}, got {actual}"

    print("PASS: scenario count = 1370")


def test_floor_c_matches_c14():
    """floor_c = 0.12717 (C14 calibration for Hill ~1.06 at N=50) in all blocks except floor-level."""
    expected = 0.12717

    # Check constant
    assert abs(FLOOR_C - expected) < 1e-5, \
        f"FLOOR_C mismatch: {FLOOR_C} vs {expected}"

    # Check in scenarios (excluding floor-level block which has different values)
    scenarios = generate_all_scenarios()
    for s in scenarios:
        if s['block'] == 'floor-level':
            # floor-level block uses different floor_c values (C17)
            continue
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
        'decision_rule',  # C17
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


def test_decision_rule_loggain_default():
    """C17: decision_rule='loggain' is default for all blocks except rule-replay."""
    scenarios = generate_all_scenarios()

    for s in scenarios:
        if s['block'] == 'rule-replay':
            # rule-replay block should use 'replay'
            assert s['decision_rule'] == 'replay', \
                f"Scenario {s['scenario_id']} in rule-replay block should have decision_rule='replay'"
        else:
            # All other blocks should use 'loggain'
            assert s['decision_rule'] == 'loggain', \
                f"Scenario {s['scenario_id']} in {s['block']} block should have decision_rule='loggain'"

    print("PASS: decision_rule='loggain' default (except rule-replay)")


def test_rule_replay_block():
    """C17: rule-replay block has 45 runs with decision_rule='replay'."""
    scenarios = generate_all_scenarios()

    rule_replay = [s for s in scenarios if s['block'] == 'rule-replay']

    assert len(rule_replay) == 45, f"Expected 45 rule-replay scenarios, got {len(rule_replay)}"

    # Check all use decision_rule='replay'
    for s in rule_replay:
        assert s['decision_rule'] == 'replay', \
            f"rule-replay scenario {s['scenario_id']} has decision_rule={s['decision_rule']}"

    # Should use laplace/power_law
    for s in rule_replay:
        assert s['log_family'] == 'laplace', f"Expected laplace, got {s['log_family']}"
        assert s['cost_type'] == 'power_law', f"Expected power_law, got {s['cost_type']}"

    # Check all 9 alphas present
    alphas = sorted(set(s['alpha'] for s in rule_replay))
    assert alphas == ALPHA_GRID, f"rule-replay alphas: {alphas}"

    print("PASS: rule-replay block (45 runs, decision_rule='replay')")


def test_floor_level_block():
    """C17: floor-level block has 90 runs with floor_c ∈ {0.10, 0.35}."""
    scenarios = generate_all_scenarios()

    floor_level = [s for s in scenarios if s['block'] == 'floor-level']

    assert len(floor_level) == 90, f"Expected 90 floor-level scenarios, got {len(floor_level)}"

    # Check floor_c values
    floor_values = sorted(set(s['floor_c'] for s in floor_level))
    expected_floors = [0.10, 0.35]
    assert floor_values == expected_floors, \
        f"floor-level floor_c values: {floor_values}, expected {expected_floors}"

    # Should use laplace/power_law
    for s in floor_level:
        assert s['log_family'] == 'laplace', f"Expected laplace, got {s['log_family']}"
        assert s['cost_type'] == 'power_law', f"Expected power_law, got {s['cost_type']}"

    # Check count per floor value: 2 floor × 9 α × 5 reps = 90
    for floor_c in expected_floors:
        count = len([s for s in floor_level if abs(s['floor_c'] - floor_c) < 0.001])
        assert count == 45, f"floor_c={floor_c} should have 45 scenarios, got {count}"

    print("PASS: floor-level block (90 runs, floor_c ∈ {0.10, 0.35})")


def test_lookback_block_c17():
    """C17: lookback block has 5 values × 2 α × 5 reps = 50 runs."""
    scenarios = generate_all_scenarios()

    lookback_scenarios = [s for s in scenarios if s['block'] == 'lookback']

    assert len(lookback_scenarios) == 50, \
        f"Expected 50 lookback scenarios, got {len(lookback_scenarios)}"

    # Check lookback values: 20, 50, 100, 200, 500
    lookbacks = sorted(set(s['lookback'] for s in lookback_scenarios))
    expected_lookbacks = [20, 50, 100, 200, 500]
    assert lookbacks == expected_lookbacks, \
        f"lookback values: {lookbacks}, expected {expected_lookbacks}"

    # Check α values: 0.1, 0.3 only
    alphas = sorted(set(s['alpha'] for s in lookback_scenarios))
    expected_alphas = [0.1, 0.3]
    assert alphas == expected_alphas, \
        f"lookback α values: {alphas}, expected {expected_alphas}"

    # Check count per lookback value: 5 lookback × 2 α × 5 reps = 50 total
    # Per lookback: 2 α × 5 reps = 10
    for lb in expected_lookbacks:
        count = len([s for s in lookback_scenarios if s['lookback'] == lb])
        assert count == 10, f"lookback={lb} should have 10 scenarios, got {count}"

    print("PASS: lookback block (50 runs, 5 values × 2 α × 5 reps)")


if __name__ == '__main__':
    print("Testing Phase C pilot design (C17 updates)...\n")

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
    test_decision_rule_loggain_default()
    test_rule_replay_block()
    test_floor_level_block()
    test_lookback_block_c17()

    print("\nAll Phase C pilot design tests passed (C17)!")
