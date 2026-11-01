# C33: Burn-in on the frozen model

Supersedes `burn_in_c.md` (C18). Model: commit `02ce733`.

The stop below is resolved by the extension to T = 16000: see `burn_in_c33_ext.md`.

## Setup

**Parameters** (from `pilot_design.make_base_params()` and the main block):
- M = N = 50
- g = 0.055, lookback = 50, merge_thresh = 0.05
- floor_c = 0.12717, sigma_range = (0.1, 0.3)
- decision_rule = loggain, market_size_fixed = True, sharing_rule = proportional
- power_law cost: c0 = 2.032e-05, c1 = 1.2
- exit_review_every at its default (1), as in every pilot run
- T = 11000, metric_every = 100, 5 replications with the pilot seeds

20 of 20 runs reproduce the stored pilot result exactly (Hill, within-market HHI, capital share and floor hits at every snapshot; K_mean and merger and exit rates in the summary).

| Cell | Scenario ids | Seeds |
|------|--------------|-------|
| laplace, α = 0 | 315-319 | 742-746 |
| laplace, α = 0.1 | 335-339 | 742-746 |
| laplace, α = 0.3 | 345-349 | 742-746 |
| t3, α = 0.3 | 525-529 | 1142-1146 |

## Definitions

- **Convergence**: first step from which the 500-step rolling mean stays within 2% of the final-2000-step mean to the end of the run, on the mean across reps (C18 criterion). "never" means no such step exists. "Max dev." is the largest deviation of the rolling mean from the final mean at steps ≥ 8000.
- **Geweke**: `burnin_check.geweke_test` on steps 8100-11000 (30 observations): first 10% (3 observations) against last 50% (15 observations), i.i.d. variance, pass if |z| < 1.96. "Rep" counts passes across 5 reps. The variance ignores autocorrelation, so the test over-rejects on persistent series, and with 33 cell-level tests about 2 failures are expected at the 5% level under stationarity.
- **Floor-hit rates**: hits by status divided by all firm-periods, as `summary['floor_hit_rate_*']` and `tidy.csv`; the two rates add up to the total rate.
- **n/a**: the series is absent or identically zero in the cell.

## Verdict

**STOP: headline series not converged by the burn-in of 8000** (K, capital share, floor-hit rates):

- laplace, α = 0.3, Cong capital share: convergence step 10500, max dev. 2.1%
- t3, α = 0.3, Cong capital share: convergence step 9100, max dev. 2.2%
- t3, α = 0.3, Floor-hit rate (standalone): convergence step 8200, max dev. 2.6%
- t3, α = 0.3, Floor-hit rate (member): convergence step 9200, max dev. 2.2%

Other series not converged by step 8000 under the 2% criterion:

- laplace, α = 0, Pooled Hill exponent: convergence step 10500, max dev. 2.4%
- laplace, α = 0, Within-market HHI: convergence step never, max dev. 4.0%
- laplace, α = 0.1, Pooled Hill exponent: convergence step never, max dev. 2.6%
- laplace, α = 0.1, Voluntary exits/period: convergence step 10100, max dev. 3.2%
- laplace, α = 0.1, Within-market HHI: convergence step 10900, max dev. 3.9%
- laplace, α = 0.3, Pooled Hill exponent: convergence step 10500, max dev. 2.2%
- laplace, α = 0.3, Voluntary exits/period: convergence step never, max dev. 4.8%
- laplace, α = 0.3, Within-market HHI: convergence step 10900, max dev. 4.2%
- t3, α = 0.3, Mergers/period: convergence step 9200, max dev. 2.2%
- t3, α = 0.3, Voluntary exits/period: convergence step never, max dev. 5.2%
- t3, α = 0.3, Floor exits/period: convergence step 9200, max dev. 2.2%
- t3, α = 0.3, Within-market HHI: convergence step never, max dev. 6.2%

Post-8000 Geweke failures on the cell mean:

- laplace, α = 0.1, Cong capital share: z = 2.60, rep passes 4/5
- t3, α = 0.3, Cong capital share: z = 2.40, rep passes 4/5
- t3, α = 0.3, Within-market HHI: z = -2.72, rep passes 3/5

## Convergence Results

### laplace, alpha=0

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | 10500 | 1.0698 | NO | 2.4% | 0.43 | pass | 4/5 |
| Mean K | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| K_eff/K | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Cong capital share | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Floor-hit rate (standalone) | 600 | 0.1153 | yes | 1.3% | -1.39 | pass | 5/5 |
| Floor-hit rate (member) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Mergers/period | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Voluntary exits/period | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Floor exits/period | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Within-market HHI | never | 0.1795 | NO | 4.0% | -0.98 | pass | 4/5 |

### laplace, alpha=0.1

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | never | 1.0916 | NO | 2.6% | 0.59 | pass | 4/5 |
| Mean K | 500 | 3.00 | yes | 1.1% | 1.13 | pass | 4/5 |
| K_eff/K | 500 | 0.7582 | yes | 0.6% | 0.38 | pass | 5/5 |
| Cong capital share | 4200 | 0.6254 | yes | 1.8% | 2.60 | FAIL | 4/5 |
| Floor-hit rate (standalone) | 600 | 0.0881 | yes | 1.4% | -1.38 | pass | 5/5 |
| Floor-hit rate (member) | 7000 | 0.0192 | yes | 1.4% | -1.45 | pass | 5/5 |
| Mergers/period | 600 | 48.2127 | yes | 1.2% | -1.26 | pass | 4/5 |
| Voluntary exits/period | 10100 | 1.8591 | NO | 3.2% | 0.19 | pass | 5/5 |
| Floor exits/period | 600 | 46.3479 | yes | 1.3% | -1.34 | pass | 5/5 |
| Within-market HHI | 10900 | 0.1738 | NO | 3.9% | -1.25 | pass | 4/5 |

### laplace, alpha=0.3

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | 10500 | 1.1652 | NO | 2.2% | 0.13 | pass | 4/5 |
| Mean K | 500 | 4.03 | yes | 1.1% | 0.96 | pass | 4/5 |
| K_eff/K | 500 | 0.7216 | yes | 0.7% | -1.65 | pass | 4/5 |
| Cong capital share | 10500 | 0.7811 | NO | 2.1% | 1.59 | pass | 4/5 |
| Floor-hit rate (standalone) | 3500 | 0.0729 | yes | 1.9% | -1.00 | pass | 5/5 |
| Floor-hit rate (member) | 3700 | 0.0151 | yes | 1.7% | -0.56 | pass | 5/5 |
| Mergers/period | 600 | 37.4646 | yes | 1.7% | -0.45 | pass | 5/5 |
| Voluntary exits/period | never | 0.9091 | NO | 4.8% | -0.34 | pass | 5/5 |
| Floor exits/period | 3700 | 36.5430 | yes | 1.6% | -0.51 | pass | 5/5 |
| Within-market HHI | 10900 | 0.1535 | NO | 4.2% | -0.97 | pass | 4/5 |

### t3, alpha=0.3

| Metric | Convergence Step | Final Mean | Before 8000 | Max dev. | Geweke z | Geweke | Rep passes |
|--------|-----------------|------------|-------------|----------|----------|--------|------------|
| Pooled Hill exponent | 8000 | 1.0862 | yes | 2.0% | 0.97 | pass | 5/5 |
| Mean K | 500 | 3.65 | yes | 1.7% | 0.85 | pass | 4/5 |
| K_eff/K | 500 | 0.7291 | yes | 0.6% | -0.79 | pass | 5/5 |
| Cong capital share | 9100 | 0.7726 | NO | 2.2% | 2.40 | FAIL | 4/5 |
| Floor-hit rate (standalone) | 8200 | 0.0853 | NO | 2.6% | -0.22 | pass | 4/5 |
| Floor-hit rate (member) | 9200 | 0.0179 | NO | 2.2% | 0.87 | pass | 4/5 |
| Mergers/period | 9200 | 43.9185 | NO | 2.2% | 0.68 | pass | 4/5 |
| Voluntary exits/period | never | 0.9440 | NO | 5.2% | -1.07 | pass | 4/5 |
| Floor exits/period | 9200 | 42.9580 | NO | 2.2% | 0.88 | pass | 4/5 |
| Within-market HHI | never | 0.1779 | NO | 6.2% | -2.72 | FAIL | 3/5 |

## Metric Tables (every 1000 steps)

Means across reps at each step.

### laplace, alpha=0

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 1000 | 1.0387 | n/a | n/a | 0.0000 | 0.1139 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1865 |
| 2000 | 1.0989 | n/a | n/a | 0.0000 | 0.1120 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1786 |
| 3000 | 1.0240 | n/a | n/a | 0.0000 | 0.1158 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1953 |
| 4000 | 1.0672 | n/a | n/a | 0.0000 | 0.1157 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1838 |
| 5000 | 1.0647 | n/a | n/a | 0.0000 | 0.1143 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1729 |
| 6000 | 1.0793 | n/a | n/a | 0.0000 | 0.1133 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1684 |
| 7000 | 1.1019 | n/a | n/a | 0.0000 | 0.1145 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1757 |
| 8000 | 1.0469 | n/a | n/a | 0.0000 | 0.1175 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1881 |
| 9000 | 1.0855 | n/a | n/a | 0.0000 | 0.1138 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1687 |
| 10000 | 1.0326 | n/a | n/a | 0.0000 | 0.1166 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1859 |
| 11000 | 1.1250 | n/a | n/a | 0.0000 | 0.1107 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1683 |

### laplace, alpha=0.1

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 1000 | 1.0512 | 2.98 | 0.7561 | 0.6149 | 0.0870 | 0.0189 | 47.1560 | 1.9020 | 45.6980 | 0.1798 |
| 2000 | 1.1228 | 2.97 | 0.7620 | 0.6474 | 0.0852 | 0.0189 | 47.8540 | 1.8500 | 45.8360 | 0.1732 |
| 3000 | 1.0569 | 2.99 | 0.7642 | 0.6417 | 0.0890 | 0.0190 | 47.7940 | 1.8000 | 46.0040 | 0.1876 |
| 4000 | 1.1111 | 2.98 | 0.7625 | 0.6446 | 0.0879 | 0.0195 | 49.1100 | 1.7980 | 47.2480 | 0.1776 |
| 5000 | 1.0922 | 2.96 | 0.7646 | 0.6180 | 0.0873 | 0.0189 | 47.3160 | 1.9520 | 45.7020 | 0.1671 |
| 6000 | 1.1272 | 3.01 | 0.7576 | 0.6243 | 0.0863 | 0.0190 | 47.7880 | 1.8140 | 45.8840 | 0.1633 |
| 7000 | 1.1168 | 3.03 | 0.7636 | 0.6156 | 0.0875 | 0.0190 | 47.9600 | 1.7540 | 45.9660 | 0.1695 |
| 8000 | 1.0807 | 2.98 | 0.7583 | 0.6304 | 0.0897 | 0.0198 | 49.3040 | 1.7280 | 47.7580 | 0.1835 |
| 9000 | 1.1174 | 2.97 | 0.7591 | 0.6624 | 0.0867 | 0.0189 | 47.4640 | 1.8760 | 45.7720 | 0.1633 |
| 10000 | 1.0651 | 2.99 | 0.7571 | 0.6274 | 0.0891 | 0.0194 | 48.6620 | 1.9220 | 46.8940 | 0.1780 |
| 11000 | 1.1591 | 3.03 | 0.7563 | 0.6440 | 0.0843 | 0.0185 | 46.9160 | 1.9200 | 44.7440 | 0.1617 |

### laplace, alpha=0.3

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 1000 | 1.1208 | 4.03 | 0.7192 | 0.7952 | 0.0724 | 0.0151 | 37.0860 | 0.9260 | 36.3820 | 0.1622 |
| 2000 | 1.2282 | 3.94 | 0.7272 | 0.7923 | 0.0696 | 0.0149 | 36.9960 | 0.8960 | 35.9600 | 0.1512 |
| 3000 | 1.1321 | 4.06 | 0.7196 | 0.7911 | 0.0736 | 0.0155 | 38.3360 | 1.0180 | 37.4240 | 0.1631 |
| 4000 | 1.1768 | 4.00 | 0.7172 | 0.7804 | 0.0727 | 0.0153 | 37.5560 | 1.0060 | 36.9140 | 0.1576 |
| 5000 | 1.1847 | 4.04 | 0.7241 | 0.7959 | 0.0729 | 0.0150 | 37.1020 | 0.8760 | 36.2540 | 0.1508 |
| 6000 | 1.1854 | 4.06 | 0.7179 | 0.7861 | 0.0703 | 0.0146 | 36.2500 | 0.8940 | 35.2120 | 0.1415 |
| 7000 | 1.1999 | 3.98 | 0.7216 | 0.7908 | 0.0718 | 0.0148 | 36.9500 | 0.9980 | 35.8140 | 0.1432 |
| 8000 | 1.1734 | 4.06 | 0.7167 | 0.7800 | 0.0735 | 0.0153 | 37.7220 | 0.8680 | 37.0340 | 0.1588 |
| 9000 | 1.1937 | 4.09 | 0.7160 | 0.7714 | 0.0714 | 0.0148 | 36.7260 | 0.9320 | 35.7900 | 0.1457 |
| 10000 | 1.1326 | 4.01 | 0.7228 | 0.7622 | 0.0732 | 0.0153 | 37.4560 | 0.9560 | 36.8380 | 0.1520 |
| 11000 | 1.2180 | 4.02 | 0.7256 | 0.8089 | 0.0694 | 0.0146 | 36.4760 | 0.8980 | 35.2880 | 0.1428 |

### t3, alpha=0.3

| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |
|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|
| 1000 | 1.0573 | 3.61 | 0.7317 | 0.7542 | 0.0875 | 0.0183 | 44.7220 | 0.9200 | 43.8860 | 0.1772 |
| 2000 | 1.1253 | 3.70 | 0.7246 | 0.7896 | 0.0816 | 0.0170 | 41.7440 | 0.8940 | 40.7920 | 0.1645 |
| 3000 | 1.0822 | 3.64 | 0.7303 | 0.7749 | 0.0874 | 0.0184 | 45.4080 | 0.9560 | 44.0260 | 0.1887 |
| 4000 | 1.0692 | 3.61 | 0.7294 | 0.7641 | 0.0872 | 0.0185 | 45.0160 | 0.8400 | 44.3680 | 0.1831 |
| 5000 | 1.0764 | 3.58 | 0.7357 | 0.7625 | 0.0853 | 0.0179 | 43.5040 | 0.8120 | 42.8680 | 0.1786 |
| 6000 | 1.0709 | 3.71 | 0.7300 | 0.7613 | 0.0842 | 0.0179 | 43.8820 | 0.9020 | 42.8580 | 0.1847 |
| 7000 | 1.0538 | 3.60 | 0.7366 | 0.7686 | 0.0861 | 0.0181 | 44.7060 | 0.9440 | 43.4940 | 0.1692 |
| 8000 | 1.0748 | 3.64 | 0.7348 | 0.7902 | 0.0834 | 0.0176 | 43.3440 | 0.9060 | 42.4500 | 0.1745 |
| 9000 | 1.0795 | 3.71 | 0.7267 | 0.8029 | 0.0852 | 0.0177 | 43.1300 | 1.0240 | 42.3240 | 0.1767 |
| 10000 | 1.1015 | 3.68 | 0.7334 | 0.7648 | 0.0864 | 0.0183 | 44.6080 | 0.9000 | 43.7840 | 0.1781 |
| 11000 | 1.1207 | 3.75 | 0.7273 | 0.7952 | 0.0798 | 0.0171 | 42.1560 | 0.9640 | 41.0840 | 0.1625 |

## Figures

- `figures/burnin_c33_laplace_a0.png`: all series for laplace, α = 0
- `figures/burnin_c33_laplace_a0.1.png`: all series for laplace, α = 0.1
- `figures/burnin_c33_laplace_a0.3.png`: all series for laplace, α = 0.3
- `figures/burnin_c33_t3_a0.3.png`: all series for t3, α = 0.3
