# Referee Fixes Mapping

This document maps Reviewer #1 items to the commits that address them on the `referee-fixes` branch.

## Summary Table

| Reviewer Item | Description | Commit | Section |
|---------------|-------------|--------|---------|
| Analytical #1 | Shock-to-firm mapping transposition | `0b31115` | §2a |
| Analytical #2 | Common random numbers across α values | `732ddcd` | §2f |
| Merger #1 | Remove α=0 merger suppression | `c91e06e` | §2c |
| Merger #2 | Per-member merger acceptance test | `d76ea51` | §2d |
| Merger #3 | Proposals counter (track merger activity) | `c91e06e` | §2c |
| Calibration #1 | Market correlation identity default | `1af30a7` | §1b/§2b |
| Calibration #2 | Seeded random correlation option | `1af30a7` | §1b/§2b |
| Calibration #3 | Cost function perturbations | `f41e29b` | §2e |
| Calibration #4 | Cost table documentation | `f41e29b` | §2e |
| Minor #1 | np.trapz → np.trapezoid (NumPy 2.0) | `c8f6f77` | §0 |
| Minor #2 | Numba cache=True for faster startup | `c8f6f77` | §0 |
| Numba-seed bug | Seed Numba RNG for reproducibility | `70f8407` | §1a |

## Detailed Commit Log

### §0 Environment: np.trapezoid and cache=True (`c8f6f77`)
- Replaced deprecated `np.trapz` with `np.trapezoid` (NumPy 2.0+ compatibility)
- Added `cache=True` to `@nb.njit` decorators for faster startup

### §1a Reproducibility: Seed Numba RNG (`70f8407`)
- Added `seed_numba(s)` function that calls `np.random.seed(s)` inside `@njit`
- Numba maintains separate per-thread RNG state that needs explicit seeding
- Called after standard `np.random.seed()` in experiment runner

### §1b/§2b Market correlation (`1af30a7`)
- Changed default correlation matrix from random to identity
- Added `--market_corr` CLI argument: "identity" (default) or "random"
- When "random", correlation matrix seeded from same seed for reproducibility

### §2a Fix transposed shock-to-firm mapping (`0b31115`)
- Changed `.T.flatten()` to `.ravel()` for row-major order
- Shocks now correctly map to firms within each market

### §2c Remove α=0 merger suppression (`c91e06e`)
- Removed the `if share > 0:` check that blocked mergers when α=0
- Added `proposals_per_period` counter to track merger activity
- Enables proper counterfactual comparison at α=0 baseline

### §2d Per-member merger acceptance test (`d76ea51`)
- Changed merger acceptance from conglomerate-level to per-member test
- Each member's synthetic log-growth must exceed realized log-growth
- Reflects that each firm must consent to the merger

### §2e Cost calibration (`f41e29b`)
- Changed from absolute parameter grids to multiplicative perturbations
- Level multipliers: [0.5, 1.0, 2.0] applied to c0
- Shape multipliers: [0.8, 1.0, 1.25] applied to c1/c2
- Generated `robustness_cost_table.md` with 84 scenarios

### §2f Common random numbers (`732ddcd`)
- Changed seed scheme: `hash((scenario_name, exp_id)) % 2**32`
- Same experiment ID uses same base seed across α values
- Enables paired comparisons that reduce variance

### §3 Performance: Vectorize conglomerate stats (`ac20076`)
- Replaced per-conglomerate Python loop with `np.bincount`
- O(n_firms) instead of O(n_conglomerates × n_firms)
- Maintains bit-identical output

### Tests (`aadd0eb`)
- `test_metrics_equivalence.py`: Verifies bincount matches loop output
- `test_shock_mapping.py`: Verifies ravel() produces correct row-major order
- `test_reproducibility.py`: Verifies seeds produce identical runs
- `test_merger_rule.py`: Verifies per-member acceptance logic

## Verification

Run tests to verify changes:
```bash
cd /groups/m-larch/bt307958/conglomerates_dev
module load python/3.13.3
source cong_env/bin/activate
python -m pytest tests/ -v
```
