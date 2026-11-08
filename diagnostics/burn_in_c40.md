# C40: Burn-in under the moment rule (pilot D), T = 16000

The C33-ext check (`burn_in_c33_ext.md`, loggain, pilot C) repeated on the pilot D economy: the same 20 scenarios (Laplace α ∈ {0, 0.1, 0.3} and t3 α = 0.3, main block, power law, proportional sharing, five reps, the pilot D seeds), `decision_rule = 'moment'`, run to T = 16000. Script `run_burnin_c40.py`.

20 of 20 runs reproduce their stored pilot D result (`pilot_d/results/scenario_XXXX.pkl.gz`) up to step 11000 exactly (Hill, within-market HHI, pooled capital share, floor hits by status).

α = 0 check: no units form under either rule and the pilot D scenarios are the moment twins of the pilot C ones (same seeds), so the α = 0 series must equal C33-ext's. All 10 series are identical (largest absolute difference 0).

## Decision rule (fixed before the runs)

burn_in = 8000 stands if the mean of each headline series over (11000, 16000] is within 2% of its mean over the pilot window (8000, 11000] in every cell, and the trend of the t3 pooled capital share over (8000, 16000] implies a change of less than 2% over those 8000 steps. Headline series: mean K, pooled capital share, floor-hit rate standalone and member (the C33 set) and the Hill exponent. Voluntary exits per period are reported, not tested: they are about 0 at α = 0.3 and a few dozen events per window at α = 0.1, so a 2% band on them measures Poisson noise.

**Result: burn_in = 8000 stands.** 17 headline means, largest difference -0.54% (Pooled Hill exponent, laplace, α = 0.1). t3 pooled capital-share trend: +0.020% per 1000 steps (p = 0.570), implied change +0.16%. On the C33 headline set alone (without Hill) the rule also holds.

## Reading

The moment-rule economy relaxes by 8,000 periods under the C33 criterion: all 17 headline means over 11,000–16,000 are within 0.54% of the pilot window (C33 under loggain: within 0.71% on its 13), and the t3 pooled-share trend implies +0.16% over 8,000 periods (loggain: −0.4%). Relaxation is not slower than under loggain. Under both rules mean K and the pooled share reach their long-run level (within 2% of the mean over 8,000–16,000) at step 500–600, the first step the 500-step rolling window can measure, although the moment economy settles at larger units (K 5.4 against 4.0 at Laplace α = 0.3) and a higher pooled share (0.91 against 0.78). The C18 band steps differ cell by cell in both directions: the pooled share stays in the band from step 500 under the moment rule in every cell against 7,100–13,300 under loggain, and mean K from 4,900 against 12,600 at Laplace α = 0.3 but from 8,000 against 500 at t3 α = 0.3. These steps measure how often a series near its level strays outside a 2% band, which depends on its noise relative to its level (the pooled share is higher and relatively steadier under the moment rule), not on how fast it got there; floor-hit rates give 12,700 in most cells under both rules. The Hill exponent never stays in the band at Laplace α = 0.3 under the moment rule (14,200 under loggain); its window means differ by −0.28% and its stationary surrogates pass the band test by 8,000 in 1% of draws, so this is its noise, as in C33.

## Headline series

Difference: (11000, 16000] against (8000, 11000], % of the pilot-window mean; 95% CI of the difference across 5 reps, in % points. Trend: OLS over (8000, 16000], rep by rep. Placebo: 300 stationary surrogates block-resampled from each rep's post-8000 window. Same functions as `burn_in_c33_ext.md`.

| Cell | Series | Pilot window | Periods 11,000–16,000 | Difference, % of pilot mean | 95% CI of the difference, % points | Trend %/1000 | p | Implied change % | Surrogates passing | P(dev ≥ observed) |
|------|--------|--------------|-----------------------|-----------------------------|------------------------------------|--------------|---|------------------|--------------------|-------------------|
| laplace, α = 0 | Pooled Hill exponent | 1.0710 | 1.0653 | -0.53 | [-2.73, +1.66] | -0.154 | 0.298 | -1.23 | 0.00 | 0.28 |
| laplace, α = 0 | Floor-hit rate (standalone) | 0.1151 | 0.1152 | +0.06 | [-1.48, +1.59] | +0.056 | 0.593 | +0.45 | 0.61 | 0.03 |
| laplace, α = 0.1 | Pooled Hill exponent | 1.1037 | 1.0978 | -0.54 | [-2.63, +1.55] | -0.079 | 0.600 | -0.64 | 0.00 | 0.53 |
| laplace, α = 0.1 | Mean K | 3.7061 | 3.6973 | -0.24 | [-0.72, +0.24] | -0.053 | 0.281 | -0.42 | 0.99 | 0.80 |
| laplace, α = 0.1 | Cong capital share | 0.8521 | 0.8529 | +0.10 | [-1.32, +1.52] | -0.001 | 0.994 | -0.01 | 0.95 | 0.30 |
| laplace, α = 0.1 | Floor-hit rate (standalone) | 0.0865 | 0.0865 | +0.02 | [-1.60, +1.64] | +0.059 | 0.610 | +0.47 | 0.37 | 0.09 |
| laplace, α = 0.1 | Floor-hit rate (member) | 0.0193 | 0.0193 | +0.12 | [-1.64, +1.88] | +0.058 | 0.583 | +0.47 | 0.47 | 0.17 |
| laplace, α = 0.3 | Pooled Hill exponent | 1.1994 | 1.1960 | -0.28 | [-1.72, +1.16] | -0.082 | 0.471 | -0.66 | 0.01 | 0.43 |
| laplace, α = 0.3 | Mean K | 5.3869 | 5.3722 | -0.27 | [-0.88, +0.34] | -0.107 | 0.286 | -0.85 | 0.76 | 0.30 |
| laplace, α = 0.3 | Cong capital share | 0.9146 | 0.9135 | -0.12 | [-0.36, +0.11] | -0.018 | 0.458 | -0.14 | 1.00 | 0.13 |
| laplace, α = 0.3 | Floor-hit rate (standalone) | 0.0697 | 0.0696 | -0.01 | [-2.08, +2.06] | +0.039 | 0.780 | +0.31 | 0.10 | 0.32 |
| laplace, α = 0.3 | Floor-hit rate (member) | 0.0148 | 0.0148 | +0.18 | [-1.97, +2.33] | +0.075 | 0.581 | +0.60 | 0.15 | 0.30 |
| t3, α = 0.3 | Pooled Hill exponent | 1.1169 | 1.1168 | -0.00 | [-1.76, +1.75] | -0.007 | 0.951 | -0.05 | 0.02 | 0.94 |
| t3, α = 0.3 | Mean K | 4.8312 | 4.8258 | -0.11 | [-1.20, +0.98] | -0.031 | 0.542 | -0.25 | 0.59 | 0.84 |
| t3, α = 0.3 | Cong capital share | 0.8927 | 0.8944 | +0.18 | [-0.39, +0.76] | +0.020 | 0.570 | +0.16 | 1.00 | 0.34 |
| t3, α = 0.3 | Floor-hit rate (standalone) | 0.0821 | 0.0816 | -0.50 | [-1.73, +0.72] | -0.100 | 0.314 | -0.80 | 0.23 | 0.75 |
| t3, α = 0.3 | Floor-hit rate (member) | 0.0175 | 0.0175 | -0.45 | [-1.41, +0.50] | -0.092 | 0.403 | -0.74 | 0.11 | 0.99 |

## Voluntary exits per period (reported, not tested)

| Cell | Pilot window | Periods 11,000–16,000 | Difference per period | Difference, % of pilot mean | 95% CI, % points |
|------|--------------|-----------------------|-----------------------|-----------------------------|------------------|
| laplace, α = 0 | 0 | 0 | 0 | n/a | n/a |
| laplace, α = 0.1 | 0.02600 | 0.02448 | -0.00152 | -5.85 | [-34.13, +22.43] |
| laplace, α = 0.3 | 0.00013 | 0.00032 | +0.00019 | +140.00 | [-215.02, +495.02] |
| t3, α = 0.3 | 0.00073 | 0.00072 | -0.00001 | -1.82 | [-68.31, +64.67] |

## Convergence steps, moment against loggain

C18 criterion on T = 16000, both rules: first step from which the 500-step rolling mean of the rep-mean series stays within 2% of its final-2000-step mean. Loggain: the C33-ext series (`burn_in_c33_ext.csv`), run through the same `analyse_series`. "never" means the series leaves the band again after every entry; as in C33-ext, the band test depends on run length and on how noisy the series is relative to 2%, so it is reported for completeness, and the window comparison above is the evidence.

First entry: the first step at which the same rolling mean comes within 2% of the series' mean over (8000, 16000], i.e. when the series first reaches its long-run level (500 is the first step the rolling window can measure).

| Cell | Series | Band step, moment | Band step, loggain | First entry, moment | First entry, loggain | Moment final mean | Loggain final mean |
|------|--------|-------------------|--------------------|---------------------|----------------------|-------------------|--------------------|
| laplace, α = 0 | Pooled Hill exponent | 15100 | 15100 | 600 | 600 | 1.0647 | 1.0647 |
| laplace, α = 0 | Floor-hit rate (standalone) | 12700 | 12700 | 600 | 600 | 0.1155 | 0.1155 |
| laplace, α = 0 | Within-market HHI | 15000 | 15000 | 600 | 600 | 0.1810 | 0.1810 |
| laplace, α = 0.1 | Pooled Hill exponent | 15100 | 13500 | 600 | 600 | 1.1007 | 1.0870 |
| laplace, α = 0.1 | Mean K | 600 | 500 | 600 | 500 | 3.70 | 3.00 |
| laplace, α = 0.1 | K_eff/K | 500 | 500 | 500 | 500 | 0.7067 | 0.7597 |
| laplace, α = 0.1 | Cong capital share | 500 | 8900 | 500 | 600 | 0.8532 | 0.6295 |
| laplace, α = 0.1 | Floor-hit rate (standalone) | 12700 | 12700 | 600 | 600 | 0.0869 | 0.0883 |
| laplace, α = 0.1 | Floor-hit rate (member) | 12700 | 12700 | 600 | 600 | 0.0193 | 0.0192 |
| laplace, α = 0.1 | Mergers/period | 12600 | 12700 | 600 | 600 | 46.9111 | 48.2309 |
| laplace, α = 0.1 | Voluntary exits/period | never | 14800 | 1100 | 500 | 0.0254 | 1.8376 |
| laplace, α = 0.1 | Floor exits/period | 12600 | 12700 | 600 | 600 | 46.9075 | 46.4085 |
| laplace, α = 0.1 | Within-market HHI | 15000 | 15000 | 700 | 600 | 0.1721 | 0.1751 |
| laplace, α = 0.3 | Pooled Hill exponent | never | 14200 | 600 | 600 | 1.1986 | 1.1795 |
| laplace, α = 0.3 | Mean K | 4900 | 12600 | 500 | 500 | 5.37 | 4.00 |
| laplace, α = 0.3 | K_eff/K | 500 | 500 | 500 | 500 | 0.6629 | 0.7206 |
| laplace, α = 0.3 | Cong capital share | 500 | 13300 | 500 | 600 | 0.9142 | 0.7782 |
| laplace, α = 0.3 | Floor-hit rate (standalone) | 12700 | 12700 | 700 | 600 | 0.0698 | 0.0730 |
| laplace, α = 0.3 | Floor-hit rate (member) | 12700 | 13900 | 700 | 600 | 0.0149 | 0.0152 |
| laplace, α = 0.3 | Mergers/period | 12700 | 13900 | 700 | 600 | 35.9773 | 37.6754 |
| laplace, α = 0.3 | Voluntary exits/period | never | never | never | 600 | 0.0002 | 0.9265 |
| laplace, α = 0.3 | Floor exits/period | 12700 | 13900 | 600 | 600 | 35.9961 | 36.7706 |
| laplace, α = 0.3 | Within-market HHI | 15000 | 15000 | 700 | 700 | 0.1469 | 0.1540 |
| t3, α = 0.3 | Pooled Hill exponent | 8300 | 15800 | 600 | 500 | 1.1167 | 1.0931 |
| t3, α = 0.3 | Mean K | 8000 | 500 | 500 | 600 | 4.81 | 3.66 |
| t3, α = 0.3 | K_eff/K | 500 | 500 | 500 | 500 | 0.6790 | 0.7280 |
| t3, α = 0.3 | Cong capital share | 500 | 7100 | 500 | 500 | 0.8945 | 0.7771 |
| t3, α = 0.3 | Floor-hit rate (standalone) | 11500 | 8200 | 600 | 600 | 0.0817 | 0.0848 |
| t3, α = 0.3 | Floor-hit rate (member) | 6700 | 10400 | 600 | 600 | 0.0175 | 0.0178 |
| t3, α = 0.3 | Mergers/period | 6700 | 6700 | 500 | 500 | 42.0945 | 43.5855 |
| t3, α = 0.3 | Voluntary exits/period | never | never | never | 600 | 0.0007 | 0.9405 |
| t3, α = 0.3 | Floor exits/period | 6700 | 6700 | 600 | 600 | 42.0885 | 42.6317 |
| t3, α = 0.3 | Within-market HHI | never | 15900 | 600 | 600 | 0.1675 | 0.1739 |

## Convergence results (moment rule)

C18 criterion on T = 16000: 500-step rolling mean within 2% of the final-2000-step mean (steps 14100-16000). Geweke on steps 8100-16000 (80 observations): first 8 against last 40, pass if |z| < 1.96.

The band test depends on run length: a longer run gives a series more chances to leave the band, so series with no trend fail it later here than at T = 11000. The window comparison above is the evidence for the burn-in; this table is reported for completeness.

### laplace, alpha=0

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | 15100 | 1.0647 | NO | 4.0% | 0.78 | pass | 4/5 |
| Mean K | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| K_eff/K | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Cong capital share | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Floor-hit rate (standalone) | 12700 | 0.1155 | NO | 2.7% | -0.55 | pass | 5/5 |
| Floor-hit rate (member) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Mergers/period | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Voluntary exits/period | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Floor exits/period | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Within-market HHI | 15000 | 0.1810 | NO | 4.8% | -0.70 | pass | 5/5 |

### laplace, alpha=0.1

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | 15100 | 1.1007 | NO | 3.3% | 0.25 | pass | 5/5 |
| Mean K | 600 | 3.70 | yes | 1.1% | 0.87 | pass | 5/5 |
| K_eff/K | 500 | 0.7067 | yes | 1.0% | -0.51 | pass | 5/5 |
| Cong capital share | 500 | 0.8532 | yes | 1.6% | -0.77 | pass | 4/5 |
| Floor-hit rate (standalone) | 12700 | 0.0869 | NO | 2.9% | -0.52 | pass | 5/5 |
| Floor-hit rate (member) | 12700 | 0.0193 | NO | 2.5% | -1.07 | pass | 4/5 |
| Mergers/period | 12600 | 46.9111 | NO | 2.1% | -1.07 | pass | 4/5 |
| Voluntary exits/period | never | 0.0254 | NO | 24.4% | 1.38 | pass | 5/5 |
| Floor exits/period | 12600 | 46.9075 | NO | 2.4% | -1.03 | pass | 4/5 |
| Within-market HHI | 15000 | 0.1721 | NO | 5.3% | -0.76 | pass | 4/5 |

### laplace, alpha=0.3

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | never | 1.1986 | NO | 3.3% | 0.44 | pass | 5/5 |
| Mean K | 4900 | 5.37 | yes | 1.9% | 0.47 | pass | 5/5 |
| K_eff/K | 500 | 0.6629 | yes | 1.2% | -0.69 | pass | 5/5 |
| Cong capital share | 500 | 0.9142 | yes | 0.5% | 1.88 | pass | 4/5 |
| Floor-hit rate (standalone) | 12700 | 0.0698 | NO | 2.9% | -0.14 | pass | 5/5 |
| Floor-hit rate (member) | 12700 | 0.0149 | NO | 2.7% | -0.76 | pass | 5/5 |
| Mergers/period | 12700 | 35.9773 | NO | 2.4% | -0.77 | pass | 5/5 |
| Voluntary exits/period | never | 0.0002 | NO | 300.0% | -2.62 | FAIL | 5/5 |
| Floor exits/period | 12700 | 35.9961 | NO | 2.7% | -0.83 | pass | 5/5 |
| Within-market HHI | 15000 | 0.1469 | NO | 6.4% | -0.52 | pass | 4/5 |

### t3, alpha=0.3

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | 8300 | 1.1167 | NO | 2.3% | 0.75 | pass | 5/5 |
| Mean K | 8000 | 4.81 | yes | 1.6% | -0.71 | pass | 5/5 |
| K_eff/K | 500 | 0.6790 | yes | 0.9% | 2.38 | FAIL | 4/5 |
| Cong capital share | 500 | 0.8945 | yes | 0.8% | -0.46 | pass | 5/5 |
| Floor-hit rate (standalone) | 11500 | 0.0817 | NO | 2.0% | 0.93 | pass | 5/5 |
| Floor-hit rate (member) | 6700 | 0.0175 | yes | 1.6% | 1.00 | pass | 5/5 |
| Mergers/period | 6700 | 42.0945 | yes | 1.7% | 1.03 | pass | 5/5 |
| Voluntary exits/period | never | 0.0007 | NO | 128.6% | 0.12 | pass | 5/5 |
| Floor exits/period | 6700 | 42.0885 | yes | 1.6% | 0.99 | pass | 5/5 |
| Within-market HHI | never | 0.1675 | NO | 5.4% | -0.79 | pass | 4/5 |

## Metric tables (12000-16000, every 1000 steps)

Means across reps at each step.

### laplace, alpha=0

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 12000 | 1.0641 | n/a | n/a | 0.0000 | 0.1181 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1840 |
| 13000 | 1.0175 | n/a | n/a | 0.0000 | 0.1167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1849 |
| 14000 | 1.0585 | n/a | n/a | 0.0000 | 0.1146 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1735 |
| 15000 | 1.0922 | n/a | n/a | 0.0000 | 0.1170 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1828 |
| 16000 | 1.0148 | n/a | n/a | 0.0000 | 0.1172 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1954 |

### laplace, alpha=0.1

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 12000 | 1.0965 | 3.67 | 0.7115 | 0.8484 | 0.0886 | 0.0200 | 48.2440 | 0.0340 | 48.3980 | 0.1747 |
| 13000 | 1.0596 | 3.67 | 0.7060 | 0.8399 | 0.0872 | 0.0196 | 47.4820 | 0.0320 | 47.4720 | 0.1749 |
| 14000 | 1.1049 | 3.76 | 0.7035 | 0.8580 | 0.0860 | 0.0194 | 47.7120 | 0.0280 | 47.0280 | 0.1639 |
| 15000 | 1.1278 | 3.72 | 0.7111 | 0.8559 | 0.0881 | 0.0196 | 47.3840 | 0.0220 | 47.4160 | 0.1743 |
| 16000 | 1.0624 | 3.66 | 0.7062 | 0.8439 | 0.0878 | 0.0195 | 47.0380 | 0.0380 | 47.3540 | 0.1835 |

### laplace, alpha=0.3

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 12000 | 1.2433 | 5.36 | 0.6518 | 0.9128 | 0.0699 | 0.0152 | 36.4980 | 0.0000 | 36.8140 | 0.1433 |
| 13000 | 1.1449 | 5.34 | 0.6667 | 0.9120 | 0.0702 | 0.0150 | 36.5020 | 0.0000 | 36.4340 | 0.1475 |
| 14000 | 1.1714 | 5.40 | 0.6586 | 0.9186 | 0.0699 | 0.0151 | 37.3860 | 0.0020 | 36.5400 | 0.1389 |
| 15000 | 1.2177 | 5.40 | 0.6622 | 0.9142 | 0.0698 | 0.0149 | 35.5200 | 0.0000 | 36.0120 | 0.1470 |
| 16000 | 1.1558 | 5.30 | 0.6665 | 0.9093 | 0.0701 | 0.0150 | 36.0140 | 0.0000 | 36.3140 | 0.1532 |

### t3, alpha=0.3

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 12000 | 1.0972 | 4.79 | 0.6754 | 0.9038 | 0.0823 | 0.0176 | 42.5940 | 0.0000 | 42.4300 | 0.1663 |
| 13000 | 1.1466 | 4.87 | 0.6844 | 0.8963 | 0.0821 | 0.0177 | 42.5840 | 0.0000 | 42.5700 | 0.1614 |
| 14000 | 1.0866 | 4.84 | 0.6831 | 0.8930 | 0.0807 | 0.0171 | 41.0500 | 0.0000 | 41.2940 | 0.1780 |
| 15000 | 1.1597 | 4.90 | 0.6821 | 0.8968 | 0.0821 | 0.0172 | 41.4560 | 0.0000 | 41.4540 | 0.1616 |
| 16000 | 1.1133 | 4.86 | 0.6785 | 0.8918 | 0.0775 | 0.0173 | 41.5460 | 0.0000 | 41.6620 | 0.1594 |

## Figures

- `figures/burnin_c40_laplace_a0.png`: all series for laplace, α = 0, moment (blue) and loggain C33-ext (grey)
- `figures/burnin_c40_laplace_a0.1.png`: all series for laplace, α = 0.1, moment (blue) and loggain C33-ext (grey)
- `figures/burnin_c40_laplace_a0.3.png`: all series for laplace, α = 0.3, moment (blue) and loggain C33-ext (grey)
- `figures/burnin_c40_t3_a0.3.png`: all series for t3, α = 0.3, moment (blue) and loggain C33-ext (grey)
