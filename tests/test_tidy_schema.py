#!/usr/bin/env python3
"""
C23: Schema validation for tidy.csv and alpha_scatter.csv.
"""
import csv
import numpy as np


TIDY_PATH = 'pilot_c/tidy.csv'
ALPHA_SCATTER_PATH = 'pilot_c/alpha_scatter.csv'

EXPECTED_TIDY_COLUMNS = [
    'scenario_id', 'cell_id', 'block', 'log_family', 'cost_type', 'alpha',
    'rep', 'seed', 'sharing_rule', 'lookback', 'cross_corr', 'alpha_endogenous',
    'decision_rule', 'elapsed_seconds', 'ms_per_step', 'cost_multiplier',
    'floor_c', 'K_median', 'K_mean', 'K_eff_over_K_median',
    'floor_hit_rate_standalone', 'floor_hit_rate_member',
    'mergers_per_period', 'proposals_per_period', 'exits_per_period',
    'floor_exits_per_period', 'voluntary_exits_per_period', 'merge_thresh',
    'growth_gap_median', 'acceptance_rate_ss', 'acceptance_rate_sc',
    'acceptance_rate_cc', 'acceptance_rate', 'hill_exponent_median',
    'hill_exponent_p25', 'hill_exponent_p75', 'hhi_within_median',
    'hhi_within_p25', 'hhi_within_p75', 'hhi_aggregate_median',
    'hhi_aggregate_p25', 'hhi_aggregate_p75', 'top10pct_aggregate_median',
    'top10pct_aggregate_p25', 'top10pct_aggregate_p75',
    'cong_capital_share_median', 'cong_capital_share_p25',
    'cong_capital_share_p75', 'K_post_burnin_median', 'K_post_burnin_p25',
    'K_post_burnin_p75', 'K_eff_post_burnin_median', 'K_eff_post_burnin_p25',
    'K_eff_post_burnin_p75', 'alpha_adopted_median', 'alpha_adopted_mean',
    'alpha_adopted_std', 'assort_iqr',
]

EXPECTED_TOTAL_ROWS = 1540

EXPECTED_BLOCK_COUNTS = {
    'main': 540,
    'equal-split': 180,
    'cost-level': 360,
    'lookback': 100,
    'correlation': 45,
    'endogenous-alpha': 60,
    'rule-replay': 45,
    'floor-level': 90,
    'search': 120,
}

EXPECTED_LOOKBACK_BY_FAMILY = {
    'laplace': 50,
    't3': 50,
}

EXPECTED_ALPHA_SCATTER_COLUMNS = [
    'scenario_id', 'rep', 'cong_id', 'alpha_final', 'K', 'sd_iqr', 'mean_iqr'
]


def load_csv(path):
    """Load CSV and return rows and header."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    return rows, header


def test_tidy_total_rows():
    """Test that tidy.csv has expected rows."""
    rows, _ = load_csv(TIDY_PATH)
    expected = EXPECTED_TOTAL_ROWS
    assert sum(EXPECTED_BLOCK_COUNTS.values()) == expected
    assert len(rows) == expected, f"Expected {expected} rows, got {len(rows)}"
    print(f"PASS: {len(rows)} total rows")


def test_tidy_columns_exist():
    """Test that all expected columns exist in tidy.csv."""
    rows, header = load_csv(TIDY_PATH)

    missing = set(EXPECTED_TIDY_COLUMNS) - set(header)
    assert not missing, f"Missing columns: {missing}"
    print(f"PASS: All {len(EXPECTED_TIDY_COLUMNS)} expected columns exist")


def test_tidy_block_counts():
    """Test block row counts."""
    rows, _ = load_csv(TIDY_PATH)

    block_counts = {}
    for row in rows:
        block = row['block']
        block_counts[block] = block_counts.get(block, 0) + 1

    for block, expected in EXPECTED_BLOCK_COUNTS.items():
        actual = block_counts.get(block, 0)
        assert actual == expected, (
            f"Block '{block}' has {actual} rows, expected {expected}"
        )
        print(f"  {block}: {actual} rows")

    print("PASS: Block counts match")


def test_lookback_by_family():
    """Test lookback block has correct counts per family."""
    rows, _ = load_csv(TIDY_PATH)

    lookback_rows = [r for r in rows if r['block'] == 'lookback']
    family_counts = {}
    for row in lookback_rows:
        family = row['log_family']
        family_counts[family] = family_counts.get(family, 0) + 1

    for family, expected in EXPECTED_LOOKBACK_BY_FAMILY.items():
        actual = family_counts.get(family, 0)
        assert actual == expected, (
            f"Lookback family '{family}' has {actual} rows, expected {expected}"
        )
        print(f"  lookback/{family}: {actual} rows")

    print("PASS: Lookback family counts match")


def test_alpha_scatter_columns():
    """Test alpha_scatter.csv has expected columns."""
    rows, header = load_csv(ALPHA_SCATTER_PATH)

    assert header == EXPECTED_ALPHA_SCATTER_COLUMNS, (
        f"Column mismatch.\n"
        f"Expected: {EXPECTED_ALPHA_SCATTER_COLUMNS}\n"
        f"Got: {header}"
    )
    print(f"PASS: alpha_scatter.csv has correct columns")


def test_alpha_scatter_no_nan_sd_iqr():
    """Test alpha_scatter.csv has no NaN in sd_iqr."""
    rows, _ = load_csv(ALPHA_SCATTER_PATH)

    nan_count = 0
    for row in rows:
        sd_iqr = row['sd_iqr']
        if sd_iqr == '' or sd_iqr == 'nan' or (sd_iqr and np.isnan(float(sd_iqr))):
            nan_count += 1

    assert nan_count == 0, f"Found {nan_count} NaN values in sd_iqr"
    print(f"PASS: No NaN in sd_iqr ({len(rows)} rows)")


if __name__ == '__main__':
    print("Testing tidy.csv schema...")
    test_tidy_total_rows()
    test_tidy_columns_exist()
    test_tidy_block_counts()
    test_lookback_by_family()

    print("\nTesting alpha_scatter.csv schema...")
    test_alpha_scatter_columns()
    test_alpha_scatter_no_nan_sd_iqr()

    print("\nAll schema tests passed!")
