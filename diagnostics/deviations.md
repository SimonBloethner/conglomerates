# C6b Burn-in Deviations

## Summary

The C6b burn-in rerun with pooled Hill estimator **FAILS** the acceptance criterion.

**Final-2000-step pooled Hill mean**: NaN (due to numerical overflow)
**Expected range**: (0.7, 1.5)

## Root Cause

With α=0.1 (10% risk pooling), the pooled Hill estimator shows extreme concentration:

1. **Initial value**: Hill ≈ 0.65 (below acceptance range)
2. **Decline**: Hill drops to ≈ 0.01 by t≈5500
3. **Overflow**: Numerical overflow (exp of large log-states) causes NaN from t≈5800

## Comparison: α=0 vs α=0.1

The acceptance criterion (0.7-1.5) is based on the α=0 theoretical result:
- With α=0 and floor_c=0.0566, the floor creates a reflecting barrier
- Theory predicts Hill ≈ 1/(1-c) ≈ 1.053
- test_stationarity confirms: Hill(t=6000) ≈ 0.97 with α=0

With α=0.1:
- Risk pooling within conglomerates changes the distribution
- Extreme concentration develops as largest firms grow unboundedly
- Pooled Hill becomes very low (≈0.01-0.65)

## Series Data

### Pooled Hill (mean across 5 reps)
| Time | Hill |
|------|------|
| 100 | 0.6513 |
| 500 | 0.2900 |
| 1000 | 0.1726 |
| 2000 | 0.1099 |
| 3000 | 0.0683 |
| 4000 | 0.0517 |
| 5000 | 0.0376 |
| 5500 | 0.0095 |
| 5800+ | NaN |

### Conglomerate Capital Share
- t=100: 17.0%
- t=1000: 0.07%
- t=2000: 3.0%
- t=3000: 0.03%
- t=5000: 10^-55 (effectively 0)
- t=5800+: 0.0 (underflow)

### Mean K
- Stable around 2.8-3.1 throughout
- Conglomerates continue forming/dissolving but with negligible capital share

## Numerical Overflow Details

The overflow occurs because:
1. With positive drift μ ∈ (0.01, 0.1), largest firms grow exponentially
2. Over 8000 steps, log-sizes can reach log(firm_size) > 700
3. exp(700) ≈ 10^304 exceeds float64 max (≈1.8×10^308)
4. Floor enforcement calls np.exp(log_states), triggering overflow

## Recommendations

1. **Reconsider acceptance criterion**: The (0.7-1.5) range applies to α=0, not α=0.1
2. **Use per-market Hill average**: The old approach (averaging per-market Hills) gave ~0.28, avoiding the extreme concentration signal
3. **Shorten burn-in runs for α>0**: Run shorter T to avoid overflow
4. **Consider alternative metrics**: For α>0 scenarios, concentration metrics other than pooled Hill may be more stable

---

# C23 Block Count Deviations

## Summary

Block counts in tidy.csv differ from C23 spec due to pickle/scenario_id desync from scenario renumbering.

## Expected vs Actual

| Block | Expected | Actual |
|-------|----------|--------|
| Total rows | 1400 | 1395 |
| endogenous-alpha | 60 | 110 |
| floor-level | 90 | 80 |
| correlation | 45 | 0 |
| lookback t3 | 40 | 80 (both families) |

## Root Cause

Scenarios were renumbered in scenarios.json but existing pickles in pilot_c/results/ retain their **original** scenario_ids. When summarize_pilot_c.py reads pickles and looks up metadata by scenario_id, it gets wrong block assignments.

## Resolution Options

1. **Full rerun**: Delete all pickles and rerun all scenarios with new IDs
2. **Remap pickles**: Rename pickle files to match new scenario_ids
3. **Accept deviation**: Document mismatch and proceed with available data

Option 3 chosen per spec instruction to document and stop if a step cannot be completed.
