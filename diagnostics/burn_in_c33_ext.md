# C33: Burn-in extended to T = 16000

Follows `burn_in_c33.md`. Same 20 scenarios and seeds, T = 16000.
20 of 20 runs reproduce the stored pilot result up to step 11000 exactly.

## Decision rule (fixed before the runs)

burn_in = 8000 stands if the mean of each headline series over (11000, 16000] is within 2% of its mean over the pilot window (8000, 11000] in every cell, and the t3 capital-share trend over (8000, 16000] implies a change of less than 2% over those 8000 steps.

**Result: burn_in = 8000 stands.** t3 capital-share trend: -0.050% per 1000 steps (p = 0.549), implied change -0.40%.

## Headline series

Difference: (11000, 16000] against (8000, 11000], % of the pilot-window mean, 95% CI across 5 reps. Trend: OLS over (8000, 16000], rep by rep. Placebo: 300 stationary surrogates block-resampled from each rep's post-8000 window (80 observations).

| Cell | Series | Pilot window | Later | Difference % | 95% CI | Trend %/1000 | p | Implied change % | Surrogates passing | P(dev ≥ observed) |
|------|--------|--------------|-------|--------------|--------|--------------|---|------------------|--------------------|-------------------|
| laplace, α = 0 | Floor-hit rate (standalone) | 0.1151 | 0.1152 | +0.06 | [-1.48, +1.59] | +0.056 | 0.593 | +0.45 | 0.54 | 0.05 |
| laplace, α = 0.1 | Mean K | 2.9985 | 2.9973 | -0.04 | [-0.49, +0.42] | +0.016 | 0.640 | +0.13 | 1.00 | 0.45 |
| laplace, α = 0.1 | Cong capital share | 0.6270 | 0.6288 | +0.28 | [-1.64, +2.20] | +0.077 | 0.563 | +0.62 | 0.06 | 0.92 |
| laplace, α = 0.1 | Floor-hit rate (standalone) | 0.0879 | 0.0880 | +0.09 | [-1.54, +1.72] | +0.070 | 0.554 | +0.56 | 0.37 | 0.04 |
| laplace, α = 0.1 | Floor-hit rate (member) | 0.0191 | 0.0191 | +0.11 | [-1.32, +1.55] | +0.064 | 0.425 | +0.51 | 0.46 | 0.04 |
| laplace, α = 0.3 | Mean K | 4.0338 | 4.0210 | -0.32 | [-1.53, +0.90] | -0.160 | 0.151 | -1.28 | 0.75 | 0.16 |
| laplace, α = 0.3 | Cong capital share | 0.7819 | 0.7842 | +0.29 | [-0.71, +1.30] | -0.052 | 0.670 | -0.41 | 0.46 | 0.13 |
| laplace, α = 0.3 | Floor-hit rate (standalone) | 0.0727 | 0.0727 | -0.03 | [-1.92, +1.87] | +0.070 | 0.600 | +0.56 | 0.10 | 0.15 |
| laplace, α = 0.3 | Floor-hit rate (member) | 0.0151 | 0.0151 | +0.18 | [-1.42, +1.78] | +0.123 | 0.311 | +0.98 | 0.15 | 0.03 |
| t3, α = 0.3 | Mean K | 3.6649 | 3.6708 | +0.16 | [-0.57, +0.89] | -0.015 | 0.882 | -0.12 | 0.81 | 0.61 |
| t3, α = 0.3 | Cong capital share | 0.7769 | 0.7749 | -0.26 | [-1.13, +0.60] | -0.050 | 0.549 | -0.40 | 0.57 | 0.47 |
| t3, α = 0.3 | Floor-hit rate (standalone) | 0.0850 | 0.0846 | -0.43 | [-1.77, +0.90] | -0.043 | 0.715 | -0.34 | 0.19 | 0.80 |
| t3, α = 0.3 | Floor-hit rate (member) | 0.0179 | 0.0177 | -0.71 | [-2.12, +0.69] | -0.127 | 0.377 | -1.01 | 0.08 | 0.91 |

## Convergence Results

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
| Pooled Hill exponent | 13500 | 1.0870 | NO | 3.9% | 0.76 | pass | 5/5 |
| Mean K | 500 | 3.00 | yes | 1.1% | -0.72 | pass | 5/5 |
| K_eff/K | 500 | 0.7597 | yes | 0.7% | 1.10 | pass | 5/5 |
| Cong capital share | 8900 | 0.6295 | NO | 2.1% | -0.83 | pass | 4/5 |
| Floor-hit rate (standalone) | 12700 | 0.0883 | NO | 3.0% | -0.59 | pass | 5/5 |
| Floor-hit rate (member) | 12700 | 0.0192 | NO | 2.9% | -1.09 | pass | 4/5 |
| Mergers/period | 12700 | 48.2309 | NO | 2.4% | -1.33 | pass | 5/5 |
| Voluntary exits/period | 14800 | 1.8376 | NO | 5.7% | 0.14 | pass | 5/5 |
| Floor exits/period | 12700 | 46.4085 | NO | 2.8% | -1.13 | pass | 4/5 |
| Within-market HHI | 15000 | 0.1751 | NO | 5.4% | -0.73 | pass | 5/5 |

### laplace, alpha=0.3

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | 14200 | 1.1795 | NO | 3.4% | -0.73 | pass | 4/5 |
| Mean K | 12600 | 4.00 | NO | 2.1% | 1.05 | pass | 5/5 |
| K_eff/K | 500 | 0.7206 | yes | 0.6% | -0.52 | pass | 5/5 |
| Cong capital share | 13300 | 0.7782 | NO | 2.5% | 0.15 | pass | 5/5 |
| Floor-hit rate (standalone) | 12700 | 0.0730 | NO | 3.1% | -0.49 | pass | 4/5 |
| Floor-hit rate (member) | 13900 | 0.0152 | NO | 3.6% | -0.67 | pass | 5/5 |
| Mergers/period | 13900 | 37.6754 | NO | 3.1% | -0.60 | pass | 5/5 |
| Voluntary exits/period | never | 0.9265 | NO | 6.1% | 0.97 | pass | 5/5 |
| Floor exits/period | 13900 | 36.7706 | NO | 3.4% | -0.72 | pass | 5/5 |
| Within-market HHI | 15000 | 0.1540 | NO | 7.2% | -0.61 | pass | 5/5 |

### t3, alpha=0.3

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | 15800 | 1.0931 | NO | 2.1% | 1.02 | pass | 5/5 |
| Mean K | 500 | 3.66 | yes | 1.6% | 0.20 | pass | 5/5 |
| K_eff/K | 500 | 0.7280 | yes | 0.6% | -0.70 | pass | 5/5 |
| Cong capital share | 7100 | 0.7771 | yes | 1.9% | 2.34 | FAIL | 4/5 |
| Floor-hit rate (standalone) | 8200 | 0.0848 | NO | 2.0% | 0.20 | pass | 5/5 |
| Floor-hit rate (member) | 10400 | 0.0178 | NO | 2.0% | 1.17 | pass | 4/5 |
| Mergers/period | 6700 | 43.5855 | yes | 1.8% | 1.29 | pass | 4/5 |
| Voluntary exits/period | never | 0.9405 | NO | 5.4% | 0.25 | pass | 5/5 |
| Floor exits/period | 6700 | 42.6317 | yes | 1.9% | 1.19 | pass | 4/5 |
| Within-market HHI | 15900 | 0.1739 | NO | 5.7% | -1.17 | pass | 5/5 |

## Metric Tables (12000-16000, every 1000 steps)

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
| 12000 | 1.0877 | 3.00 | 0.7586 | 0.6298 | 0.0899 | 0.0197 | 49.4120 | 1.8740 | 47.5040 | 0.1777 |
| 13000 | 1.0427 | 2.98 | 0.7572 | 0.6250 | 0.0891 | 0.0194 | 48.7940 | 1.7920 | 46.9940 | 0.1780 |
| 14000 | 1.0763 | 3.02 | 0.7574 | 0.6394 | 0.0877 | 0.0192 | 48.8600 | 1.7820 | 46.3880 | 0.1666 |
| 15000 | 1.1059 | 3.00 | 0.7607 | 0.6282 | 0.0893 | 0.0193 | 48.4360 | 1.8000 | 46.7260 | 0.1760 |
| 16000 | 1.0383 | 2.99 | 0.7640 | 0.6188 | 0.0894 | 0.0194 | 48.4360 | 1.7740 | 46.8760 | 0.1880 |

### laplace, alpha=0.3

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 12000 | 1.1911 | 4.09 | 0.7244 | 0.7819 | 0.0740 | 0.0155 | 38.2040 | 0.9080 | 37.3460 | 0.1569 |
| 13000 | 1.1569 | 3.98 | 0.7243 | 0.7898 | 0.0722 | 0.0150 | 37.0460 | 1.0100 | 36.1580 | 0.1510 |
| 14000 | 1.1478 | 4.05 | 0.7183 | 0.7996 | 0.0723 | 0.0151 | 38.0700 | 0.9280 | 36.5120 | 0.1463 |
| 15000 | 1.1943 | 4.00 | 0.7221 | 0.7759 | 0.0734 | 0.0155 | 38.1500 | 0.9200 | 37.2560 | 0.1537 |
| 16000 | 1.1643 | 3.96 | 0.7251 | 0.7406 | 0.0734 | 0.0155 | 37.7820 | 0.8460 | 37.3420 | 0.1620 |

### t3, alpha=0.3

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 12000 | 1.0922 | 3.61 | 0.7291 | 0.7904 | 0.0850 | 0.0179 | 43.7940 | 0.8820 | 42.8940 | 0.1719 |
| 13000 | 1.0816 | 3.71 | 0.7316 | 0.7509 | 0.0849 | 0.0178 | 43.7940 | 0.9200 | 42.6980 | 0.1746 |
| 14000 | 1.0952 | 3.68 | 0.7221 | 0.7698 | 0.0836 | 0.0176 | 42.9340 | 0.9100 | 42.2420 | 0.1799 |
| 15000 | 1.1261 | 3.65 | 0.7271 | 0.8003 | 0.0846 | 0.0175 | 42.9580 | 1.0080 | 42.0360 | 0.1692 |
| 16000 | 1.0565 | 3.74 | 0.7279 | 0.7821 | 0.0815 | 0.0175 | 43.0100 | 0.8420 | 42.0700 | 0.1657 |

## Figures

- `figures/burnin_c33_ext_laplace_a0.png`: all series for laplace, α = 0
- `figures/burnin_c33_ext_laplace_a0.1.png`: all series for laplace, α = 0.1
- `figures/burnin_c33_ext_laplace_a0.3.png`: all series for laplace, α = 0.3
- `figures/burnin_c33_ext_t3_a0.3.png`: all series for t3, α = 0.3
