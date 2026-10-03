#!/usr/bin/env python3
"""
Test pilot design functionality (§7).

- Verifies factorial design generation
- Tests experiment table expansion
"""
import sys
sys.path.insert(0, '..')
from pilot_design import (
    generate_factorial_design,
    generate_experiment_table,
    estimate_runtime,
    FACTORS,
    ALPHA_VALUES
)


def test_factorial_design_count():
    """
    Factorial design should have correct number of scenarios.
    """
    expected_count = 1
    for factor_levels in FACTORS.values():
        expected_count *= len(factor_levels)

    scenarios = generate_factorial_design()

    assert len(scenarios) == expected_count, \
        f"Expected {expected_count} scenarios, got {len(scenarios)}"

    print(f"Factorial design: {len(scenarios)} scenarios")


def test_factorial_design_completeness():
    """
    All factor combinations should be present.
    """
    scenarios = generate_factorial_design()

    # Extract combinations
    combos = set()
    for s in scenarios:
        combo = tuple(s[f] for f in FACTORS.keys())
        combos.add(combo)

    # Should have all unique combinations
    expected_count = 1
    for factor_levels in FACTORS.values():
        expected_count *= len(factor_levels)

    assert len(combos) == expected_count, \
        f"Expected {expected_count} unique combinations, got {len(combos)}"

    print("All factor combinations present")


def test_experiment_table_expansion():
    """
    Experiment table should have correct dimensions.
    """
    scenarios = generate_factorial_design()
    n_scenarios = len(scenarios)
    n_alpha = len(ALPHA_VALUES)
    n_reps = 5

    df = generate_experiment_table(scenarios, n_reps=n_reps)

    expected_rows = n_scenarios * n_alpha * n_reps

    assert len(df) == expected_rows, \
        f"Expected {expected_rows} rows, got {len(df)}"

    # Check columns
    assert 'scenario_id' in df.columns
    assert 'alpha' in df.columns
    assert 'replication' in df.columns
    assert 'experiment_id' in df.columns

    print(f"Experiment table: {len(df)} rows × {len(df.columns)} columns")


def test_experiment_ids_unique():
    """
    All experiment IDs should be unique.
    """
    scenarios = generate_factorial_design()
    df = generate_experiment_table(scenarios, n_reps=5)

    assert df['experiment_id'].nunique() == len(df), \
        "Experiment IDs should be unique"

    print("All experiment IDs unique")


def test_alpha_values_correct():
    """
    α values should match the updated grid.
    """
    scenarios = generate_factorial_design()
    df = generate_experiment_table(scenarios, n_reps=1)

    alphas_in_table = sorted(df['alpha'].unique())

    assert len(alphas_in_table) == len(ALPHA_VALUES), \
        f"Expected {len(ALPHA_VALUES)} alpha values, got {len(alphas_in_table)}"

    for a1, a2 in zip(alphas_in_table, sorted(ALPHA_VALUES)):
        assert abs(a1 - a2) < 1e-6, f"Alpha mismatch: {a1} vs {a2}"

    print(f"Alpha values correct: {len(alphas_in_table)} values")


def test_runtime_estimate():
    """
    Runtime estimate should return positive values.
    """
    est = estimate_runtime(36, 12, 5)

    assert est['n_experiments'] == 36 * 12 * 5
    assert est['total_serial_hours'] > 0
    assert est['parallel_64cores_hours'] > 0
    assert est['parallel_64cores_hours'] < est['total_serial_hours']

    print(f"Runtime estimate: {est['total_serial_hours']:.1f}h serial, "
          f"{est['parallel_64cores_hours']:.1f}h with 64 cores")


if __name__ == '__main__':
    print("Testing factorial design count...")
    test_factorial_design_count()
    print("PASS\n")

    print("Testing factorial design completeness...")
    test_factorial_design_completeness()
    print("PASS\n")

    print("Testing experiment table expansion...")
    test_experiment_table_expansion()
    print("PASS\n")

    print("Testing experiment ID uniqueness...")
    test_experiment_ids_unique()
    print("PASS\n")

    print("Testing alpha values...")
    test_alpha_values_correct()
    print("PASS\n")

    print("Testing runtime estimate...")
    test_runtime_estimate()
    print("PASS\n")

    print("All pilot design tests passed!")
