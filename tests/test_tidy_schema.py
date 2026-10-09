#!/usr/bin/env python3
"""
Test: tidy.csv schema validation.

Verifies the full column list and block row counts.
"""
import csv
import os

TIDY_PATH = 'pilot_c/tidy.csv'

EXPECTED_COLUMNS = [
    'scenario_id', 'cell_id', 'block', 'log_family', 'cost_type', 'alpha',
    'rep', 'seed', 'sharing_rule', 'lookback', 'cross_corr', 'alpha_endogenous',
    'decision_rule', 'elapsed_seconds', 'ms_per_step', 'cost_multiplier',
    'floor_c', 'K_median', 'K_mean', 'K_eff_over_K_median',
    'floor_hit_rate_standalone', 'floor_hit_rate_member',
    'mergers_per_period', 'proposals_per_period', 'exits_per_period',
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
    'alpha_adopted_std', 'assort_iqr', 'event_n_events', 'event_n_matched',
    'event_did_median', 'event_did_p25', 'event_did_p75',
    'event_joiner_before_median', 'event_joiner_after_median',
    'event_control_before_median', 'event_control_after_median',
    'event_did_post_median',
]

EXPECTED_BLOCK_COUNTS = {
    'main': 540,
    'equal-split': 180,
    'cost-level': 360,
    'lookback': 50,
    'correlation': 45,
    'endogenous-alpha': 60,
    'rule-replay': 45,
    'floor-level': 90,
}


def load_tidy_csv():
    """Load tidy.csv and return rows and header."""
    with open(TIDY_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    return rows, header


def test_column_list():
    """Test that tidy.csv has all expected columns in order."""
    rows, header = load_tidy_csv()

    assert header == EXPECTED_COLUMNS, (
        f"Column mismatch.\n"
        f"Expected: {EXPECTED_COLUMNS}\n"
        f"Got: {header}\n"
        f"Missing: {set(EXPECTED_COLUMNS) - set(header)}\n"
        f"Extra: {set(header) - set(EXPECTED_COLUMNS)}"
    )
    print(f"PASS: {len(header)} columns match expected schema")


def test_block_row_counts():
    """Test that each block has the expected row count."""
    rows, _ = load_tidy_csv()

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

    total = sum(block_counts.values())
    expected_total = sum(EXPECTED_BLOCK_COUNTS.values())
    print(f"PASS: Block counts match ({total} total rows)")


def test_total_rows():
    """Test total row count."""
    rows, _ = load_tidy_csv()
    expected = sum(EXPECTED_BLOCK_COUNTS.values())
    assert len(rows) == expected, f"Expected {expected} rows, got {len(rows)}"
    print(f"PASS: {len(rows)} total rows")


if __name__ == '__main__':
    print("Testing tidy.csv schema...")
    test_column_list()
    test_block_row_counts()
    test_total_rows()
    print("\nAll schema tests passed!")
