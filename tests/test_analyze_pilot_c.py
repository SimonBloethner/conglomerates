#!/usr/bin/env python3
"""
Tests for Phase C pilot analysis.

Uses synthetic tidy.csv that catches wrong group-by operations.
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd

# Handle both direct execution and pytest
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from analyze_pilot_c import (
    agg_with_bands,
    generate_hill_vs_alpha_table,
    generate_floor_hit_table,
    generate_equal_split_comparison,
    generate_runtime_stats,
)


def create_synthetic_tidy():
    """
    Create synthetic tidy.csv with known values.

    Design:
    - 3 families: normal, laplace, t3
    - 2 alphas: 0.0, 0.1
    - 2 reps per cell
    - Values designed so wrong group-by gives wrong medians

    For each (family, alpha) cell:
      - rep 0: K = 10 + alpha_idx * 5 + family_idx * 1
      - rep 1: K = K + 2

    This ensures:
      - Median across reps is (K + K+2) / 2 = K + 1
      - Wrong grouping (e.g., grouping by alpha only) gives different median
    """
    rows = []
    families = ['normal', 'laplace', 't3']
    alphas = [0.0, 0.1]
    scenario_id = 0

    for family_idx, family in enumerate(families):
        for alpha_idx, alpha in enumerate(alphas):
            for rep in range(2):
                base_k = 10 + alpha_idx * 5 + family_idx * 1
                K = base_k + rep * 2

                rows.append({
                    'scenario_id': scenario_id,
                    'cell_id': family_idx * 2 + alpha_idx,
                    'block': 'main',
                    'log_family': family,
                    'cost_type': 'power_law',
                    'alpha': alpha,
                    'rep': rep,
                    'seed': 42 + scenario_id,
                    'sharing_rule': 'proportional',
                    'lookback': 50,
                    'cross_corr': 0.0,
                    'alpha_endogenous': False,
                    'cost_multiplier': 1.0,
                    'floor_c': 0.0566,
                    'elapsed_seconds': 60.0 + scenario_id,
                    'ms_per_step': 10.0 + scenario_id * 0.1,
                    # Metrics: designed for testable medians
                    'K_median': K,
                    'K_eff_over_K_median': 0.9,
                    'K_post_burnin_median': K,
                    'K_post_burnin_p25': K - 1,
                    'K_post_burnin_p75': K + 1,
                    'K_eff_post_burnin_median': K * 0.9,
                    'K_eff_post_burnin_p25': K * 0.85,
                    'K_eff_post_burnin_p75': K * 0.95,
                    'floor_hit_rate_standalone': 0.01 * (family_idx + 1),
                    'floor_hit_rate_member': 0.005 * (family_idx + 1),
                    'hill_exponent_median': 1.0 + alpha * 0.5 + family_idx * 0.1,
                    'hill_exponent_p25': 0.9 + alpha * 0.5 + family_idx * 0.1,
                    'hill_exponent_p75': 1.1 + alpha * 0.5 + family_idx * 0.1,
                    'hhi_within_median': 0.1 + alpha * 0.02,
                    'hhi_within_p25': 0.09 + alpha * 0.02,
                    'hhi_within_p75': 0.11 + alpha * 0.02,
                    'hhi_aggregate_median': 0.02 + alpha * 0.01,
                    'hhi_aggregate_p25': 0.018 + alpha * 0.01,
                    'hhi_aggregate_p75': 0.022 + alpha * 0.01,
                    'top10_aggregate_median': 0.3 + alpha * 0.05,
                    'top10_aggregate_p25': 0.28 + alpha * 0.05,
                    'top10_aggregate_p75': 0.32 + alpha * 0.05,
                    'cong_capital_share_median': 0.5,
                    'cong_capital_share_p25': 0.45,
                    'cong_capital_share_p75': 0.55,
                    'mergers_per_period': 0.1,
                    'proposals_per_period': 0.5,
                    'exits_per_period': 0.05,
                    'alpha_adopted_median': np.nan,
                    'alpha_adopted_mean': np.nan,
                    'alpha_adopted_std': np.nan,
                })
                scenario_id += 1

    return pd.DataFrame(rows)


def create_equal_split_synthetic():
    """
    Create synthetic data with main and equal-split blocks.

    Design for catching wrong block filtering:
    - main block: normal, K=10,12 (reps)
    - equal-split block: normal, K=20,22 (reps)
    - If block filter is wrong, medians will be mixed
    """
    rows = []
    scenario_id = 0

    # Main block
    for rep in range(2):
        K = 10 + rep * 2
        rows.append({
            'scenario_id': scenario_id,
            'cell_id': 0,
            'block': 'main',
            'log_family': 'normal',
            'cost_type': 'power_law',
            'alpha': 0.1,
            'rep': rep,
            'seed': 42 + scenario_id,
            'sharing_rule': 'proportional',
            'lookback': 50,
            'cross_corr': 0.0,
            'alpha_endogenous': False,
            'cost_multiplier': 1.0,
            'K_post_burnin_median': K,
            'K_post_burnin_p25': K - 1,
            'K_post_burnin_p75': K + 1,
            'elapsed_seconds': 60.0,
            'ms_per_step': 10.0,
        })
        scenario_id += 1

    # Equal-split block
    for rep in range(2):
        K = 20 + rep * 2
        rows.append({
            'scenario_id': scenario_id,
            'cell_id': 1,
            'block': 'equal-split',
            'log_family': 'normal',
            'cost_type': 'power_law',
            'alpha': 0.1,
            'rep': rep,
            'seed': 42 + scenario_id,
            'sharing_rule': 'equal',
            'lookback': 50,
            'cross_corr': 0.0,
            'alpha_endogenous': False,
            'cost_multiplier': 1.0,
            'K_post_burnin_median': K,
            'K_post_burnin_p25': K - 1,
            'K_post_burnin_p75': K + 1,
            'elapsed_seconds': 60.0,
            'ms_per_step': 10.0,
        })
        scenario_id += 1

    return pd.DataFrame(rows)


def test_agg_with_bands():
    """Test agg_with_bands produces correct statistics."""
    df = create_synthetic_tidy()

    # Aggregate K by family and alpha
    result = agg_with_bands(df, ['log_family', 'alpha'], 'K_post_burnin_median')

    # Check normal, alpha=0.0
    # base_k = 10 + 0*5 + 0*1 = 10
    # rep 0: K=10, rep 1: K=12
    # median = 11
    normal_a0 = result[(result['log_family'] == 'normal') & (result['alpha'] == 0.0)]
    assert len(normal_a0) == 1
    assert normal_a0['K_post_burnin_median_median'].iloc[0] == 11.0
    assert normal_a0['n_reps'].iloc[0] == 2

    # Check laplace, alpha=0.1
    # base_k = 10 + 1*5 + 1*1 = 16
    # rep 0: K=16, rep 1: K=18
    # median = 17
    laplace_a1 = result[(result['log_family'] == 'laplace') & (result['alpha'] == 0.1)]
    assert len(laplace_a1) == 1
    assert laplace_a1['K_post_burnin_median_median'].iloc[0] == 17.0

    print("PASS: agg_with_bands")


def test_hill_vs_alpha_groupby():
    """Test Hill table groups by (family, alpha), not just alpha."""
    df = create_synthetic_tidy()

    result = generate_hill_vs_alpha_table(df)

    # Should have 6 rows: 3 families × 2 alphas
    assert len(result) == 6, f"Expected 6 rows, got {len(result)}"

    # Check that families are distinguished
    families_at_a0 = result[result['alpha'] == 0.0]['log_family'].tolist()
    assert set(families_at_a0) == {'normal', 'laplace', 't3'}

    # Check specific values (design: hill = 1.0 + alpha*0.5 + family_idx*0.1)
    # normal, alpha=0: 1.0 + 0 + 0 = 1.0
    # laplace, alpha=0: 1.0 + 0 + 0.1 = 1.1
    # t3, alpha=0: 1.0 + 0 + 0.2 = 1.2
    normal_a0 = result[(result['log_family'] == 'normal') & (result['alpha'] == 0.0)]
    laplace_a0 = result[(result['log_family'] == 'laplace') & (result['alpha'] == 0.0)]

    assert abs(normal_a0['hill_exponent_median_median'].iloc[0] - 1.0) < 0.01
    assert abs(laplace_a0['hill_exponent_median_median'].iloc[0] - 1.1) < 0.01

    print("PASS: hill vs alpha groupby")


def test_floor_hit_groupby():
    """Test floor-hit table groups correctly."""
    df = create_synthetic_tidy()

    result = generate_floor_hit_table(df)

    # Should have 6 rows: 3 families × 2 alphas
    assert len(result) == 6, f"Expected 6 rows, got {len(result)}"

    # Design: standalone = 0.01 * (family_idx + 1)
    # normal: 0.01, laplace: 0.02, t3: 0.03
    normal_row = result[(result['log_family'] == 'normal') & (result['alpha'] == 0.0)]
    assert abs(normal_row['floor_hit_rate_standalone_median'].iloc[0] - 0.01) < 0.001

    laplace_row = result[(result['log_family'] == 'laplace') & (result['alpha'] == 0.0)]
    assert abs(laplace_row['floor_hit_rate_standalone_median'].iloc[0] - 0.02) < 0.001

    print("PASS: floor hit groupby")


def test_equal_split_comparison_filters_block():
    """Test equal-split comparison correctly filters by block."""
    df = create_equal_split_synthetic()

    result = generate_equal_split_comparison(df)

    # Should have 1 row: alpha=0.1, cost=power_law
    assert len(result) == 1, f"Expected 1 row, got {len(result)}"

    row = result.iloc[0]

    # main block: K=10,12 -> median=11
    # equal-split block: K=20,22 -> median=21
    assert row['K_main_median'] == 11.0, f"Expected K_main=11, got {row['K_main_median']}"
    assert row['K_equal_median'] == 21.0, f"Expected K_equal=21, got {row['K_equal_median']}"

    # If block filtering was wrong, these would be mixed
    print("PASS: equal split comparison filters block")


def test_runtime_stats():
    """Test runtime statistics computation."""
    df = create_synthetic_tidy()

    stats = generate_runtime_stats(df)

    assert stats['n_scenarios'] == 12  # 3 families × 2 alphas × 2 reps
    assert stats['total_seconds'] > 0
    assert stats['mean_ms_per_step'] > 0

    print("PASS: runtime stats")


def test_synthetic_csv_columns():
    """Test that synthetic CSV has required columns."""
    df = create_synthetic_tidy()

    required_cols = [
        'scenario_id', 'cell_id', 'block', 'log_family', 'cost_type', 'alpha',
        'rep', 'seed', 'sharing_rule', 'lookback', 'cross_corr', 'alpha_endogenous',
        'K_post_burnin_median', 'hill_exponent_median',
        'floor_hit_rate_standalone', 'floor_hit_rate_member',
        'hhi_within_median', 'hhi_aggregate_median', 'top10_aggregate_median',
        'elapsed_seconds', 'ms_per_step',
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    print(f"PASS: synthetic CSV has {len(required_cols)} required columns")


def test_groupby_catches_wrong_aggregation():
    """
    Test that verifies wrong group-by would fail.

    If we grouped by alpha only (ignoring family), the medians would differ.
    """
    df = create_synthetic_tidy()

    # Correct: group by family and alpha
    correct = df.groupby(['log_family', 'alpha'])['K_post_burnin_median'].median()

    # Wrong: group by alpha only
    wrong = df.groupby(['alpha'])['K_post_burnin_median'].median()

    # At alpha=0:
    # - correct: normal=11, laplace=12, t3=13
    # - wrong: median([10,11,12,12,13,14]) for all reps at alpha=0 = 12

    # The wrong aggregation gives a single value per alpha
    assert len(wrong) == 2  # 2 alphas
    assert len(correct) == 6  # 3 families × 2 alphas

    # Verify correct values differ by family
    assert correct[('normal', 0.0)] != correct[('laplace', 0.0)]
    assert correct[('laplace', 0.0)] != correct[('t3', 0.0)]

    print("PASS: groupby catches wrong aggregation")


if __name__ == '__main__':
    print("Testing Phase C pilot analysis...\n")

    test_agg_with_bands()
    test_hill_vs_alpha_groupby()
    test_floor_hit_groupby()
    test_equal_split_comparison_filters_block()
    test_runtime_stats()
    test_synthetic_csv_columns()
    test_groupby_catches_wrong_aggregation()

    print("\nAll Phase C analysis tests passed!")
