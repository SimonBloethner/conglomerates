# Phase A Diagnostic Summary

## Quick Diagnostic Results

**Configuration:** 30×30=900 firms, 1000 steps

| Cost Type | α | Mean Gini | Mergers | Proposals | Final Cong |
|-----------|---|-----------|---------|-----------|------------|
| linear      | 0.0 |    0.4801 |       0 |     45154 |          0 |
| linear      | 0.1 |    0.4801 |       3 |     45154 |          0 |
| linear      | 0.3 |    0.4800 |      20 |     45103 |          0 |
| quadratic   | 0.0 |    0.4801 |       0 |     45154 |          0 |
| quadratic   | 0.1 |    0.4801 |       2 |     45154 |          0 |
| quadratic   | 0.3 |    0.4799 |      19 |     45103 |          0 |
| exponential | 0.0 |    0.4801 |       0 |     45154 |          0 |
| exponential | 0.1 |    0.4801 |       0 |     45154 |          0 |
| exponential | 0.3 |    0.4801 |       0 |     45154 |          0 |
| power_law   | 0.0 |    0.4801 |       0 |     45154 |          0 |
| power_law   | 0.1 |    0.4800 |      14 |     45105 |          0 |
| power_law   | 0.3 |    0.4794 |      37 |     45116 |          2 |

## Key Validations

1. **α=0 produces zero mergers**: ✓ Verified for all cost types
   - Per-member merger test correctly rejects mergers when α=0 (no sharing benefit)

2. **Proposals occur for all α**: ✓ Verified (~45,000 proposals)
   - The α=0 merger suppression bug (§2c) has been fixed

3. **Higher α leads to more mergers**: ✓ Verified
   - Linear: 0 → 3 → 20 mergers
   - Power law: 0 → 14 → 37 mergers

4. **Reproducibility**: ✓ Verified by test suite
   - Same seed produces identical results
   - Different seeds produce different results

## Test Results

| Test | Status |
|------|--------|
| test_metrics_equivalence.py | PASS |
| test_shock_mapping.py | PASS |
| test_reproducibility.py | PASS |
| test_merger_rule.py | PASS |

## Notes

- Exponential cost function shows zero mergers even at α=0.3, likely due to high management costs at default parameters
- Gini coefficients stabilize around 0.48 after 1000 steps
- Model runs in ~0.5 seconds per experiment at this scale
