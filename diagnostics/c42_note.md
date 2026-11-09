# C42: rank mobility (traces) and the accept-all ceiling

- **C42a**: rank mobility and persistence from the traces (`analysis/rank_mobility.py`, `pilot_d/rank_mobility.csv`),
  and the May article's per-firm rank range and rank SD (`pilot_d/rank_range_sd.csv`), Section 1.
- **C42b**: `decision_rule='accept'`, its tests, the 45 accept runs, the identity check, Section 2 and Table 3c re-cut
  (`analysis/paper_tables.py --table3c-accept`).

No pilot C or pilot D result changes. New runs: the 45 accept runs (`pilot_d/results_c42/`, appended to
`scenarios.json` and `tidy.csv`), the nine α = 0 traces, and the 2,055 identity reruns. The 45 accept pickles are on
Festus in `IOxEE_c42`; the traces and the identity reruns are on /scratch only (README).

## 1. Rank mobility and persistence within markets

**Data.** The 27 traced main power-law runs (three families, α ∈ {0.05, 0.1, 0.3}, reps 0–2). The brief allowed
adding the α = 0 cells. A traced α = 0 run takes 2 minutes, so all three reps of each family were added
(`analysis/run_traces.py --alpha0`, nine runs, not three). Each trace is checked against its result pickle: floor
hits per step agree.

**Measures.** Periods are t > 8,000, with an origin every 100 periods. Ranks are of the log market share within the
market (50 firms), at t and t + h. The sample `hitfree` drops every firm with a floor lift in (t, t + h] and
re-ranks the rest, so a lifted firm moves neither its own rank nor anyone else's. Markets with fewer than 10
lift-free firms are skipped. Owners rank owning units across the economy: a unit counts once, and capital is the sum
of its firms' shares. An owner is followed through its largest firm. The script's docstring has the definitions.

**Answer.**
- **Pooling lowers rank mobility within markets; it does not raise it.** Every measure moves the same way as α rises
  (all firms, h = 100, α = 0 → 0.3):
  - Spearman rank persistence rises: 0.33 → 0.45 (normal), 0.21 → 0.32 (Laplace), 0.16 → 0.25 (t₃);
  - top-decile persistence rises: 0.48 → 0.53, 0.35 → 0.41, 0.31 → 0.36;
  - leader turnover falls slightly: −0.016 to −0.019;
  - the mean absolute rank change falls by 1.0–1.5 places.
  - At α = 0.05 the movement is barely visible (Spearman +0.01).
  - These are the brief's lag-based measures, not the May article's, and pilot D is not the article's model. The
    next item compares like with like.
- **The May article's measures say the same, barely.** The article (Section 5, Figure `fig:mobility_ridges`) measured
  each firm's rank range (max − min) and rank SD over the run: the model's online `rank_range` and `rank_std`.
  `--range-sd` recomputes both from the traces (`pilot_d/rank_range_sd.csv`); a test checks that they equal the
  model's online statistics.
  - **Both are at their ceiling in pilot D.** Over the 3,000 post-burn-in periods, the median firm's range is 48–49
    of a possible 49, and 51–75% of firms reach 49 (over the whole run, 91–99%). The median rank SD is 13.6–14.1,
    against 14.43 for a rank drawn uniformly from 1..50.
  - **What movement there is goes the same way as above.** From α = 0 to 0.3:
    - the mean post-burn-in rank SD falls: 13.81 → 13.54 (normal), 14.04 → 13.87 (Laplace), 14.11 → 13.99 (t₃);
    - the share of firms at the maximum range falls: 0.55 → 0.51, 0.74 → 0.70, 0.75 → 0.72.
  - **So the different sign comes from the model, not the measure.** In the article, α = 0 was pure Gibrat growth
    without a floor: leaders lock in, ranks barely move, and any pooling raises mobility. In pilot D the reflecting
    floor recapitalizes small firms (68–83% are lifted within 100 periods at α = 0), so the baseline is already
    close to maximal mobility. Pooling then slows it slightly, by damping members' growth and cutting floor hits.
  - **Not yet confirmed.** The floor is the obvious difference, but pilot D also differs in the decision rule and in
    fixing market sizes. A traced pilot D run without the floor would confirm it; that run was not made.
- **The floor drives part of that fall.** The lift-free Spearman rises by less, 0.03–0.08. So part of the fall is
  fewer floor lifts: the share of firms lifted within 100 periods falls from 0.68–0.83 at α = 0 to 0.52–0.71 at
  α = 0.3. The rest is slower movement among going concerns.
- **Mobility is high in every cell.** Within 100 periods a market's ranks are largely reshuffled (Spearman
  0.16–0.45). By h = 500 they are close to independent (0.01–0.09).
- **At α = 0.05, floor recapitalization is a modest share of mobility.** Measured as 1 − Spearman, the lift-free
  sample is 11–14% less mobile: 0.66 → 0.58 (normal), 0.78 → 0.69 (Laplace), 0.83 → 0.71 (t₃). At the bottom
  decile the floor's share is about 5% (normal: persistence 0.13 → 0.17). At the top it is none: the lift-free sample
  is no less mobile there (normal: top-decile persistence 0.46 against 0.48, leader turnover 0.62 against 0.59). Per position held, the mean rank
  change is the same in both samples: 12.9 / 50 and 4.9 / 19.
  - **Caveat.** At α = 0.05, 66–81% of firms are lifted at least once within 100 periods. The lift-free sample is
    therefore a selected minority of the larger firms, about 15–19 per market. "Mobility among going concerns" is
    mobility among those firms.
- **Members' ranks move less than standalones' in every cell.** The mean absolute rank change is 9–20% lower for
  members: at α = 0.3, 10.9 against 13.6 (normal), 12.5 against 14.9 (Laplace), 13.1 against 15.3 (t₃). This holds
  in the lift-free sample too.
  - **Top-decile persistence changes sign with α.** At α = 0.3 members persist more: 0.53 against 0.48 (normal),
    0.41 against 0.30 (Laplace), 0.37 against 0.28 (t₃). At α = 0.05 the few standalones in the top decile persist
    more: 0.65 against 0.46 (normal), 0.43 against 0.35 (Laplace).
  - The standalone rows rest on few firms: members hold 84–99% of the top decile.
  - Members are also lifted less often: 41–73% of them within 100 periods, against 82–91% of standalones.
  - So the C41 compression of both tails shows up as stickier ranks for members overall. Whether members also
    persist more at the top depends on α.
- **Across the economy, pooling makes the top of the ownership distribution more persistent.**
  - The share of the top decile of owners still in it 100 periods later rises from 0.49 / 0.37 / 0.32 at α = 0 to
    0.69 / 0.68 / 0.67 at α = 0.3.
  - Owner Spearman rises from 0.33 / 0.21 / 0.16 to 0.40 / 0.28 / 0.23.
  - The largest owner is replaced within 100 periods less often: 0.89 / 0.99 / 0.98 → 0.79 / 0.94 / 0.87.
  - This sits beside the concentration measure of 4.6: the top decile's capital share rises from 0.65 / 0.65 / 0.68
    to 0.67 / 0.73 / 0.77. The trace values match `top10pct_aggregate` from the pickles within 0.003.
  - Pooling thus concentrates ownership and makes the top owners more persistent, while ranks within markets stay
    highly mobile.

### Tables (mean over the three reps; `pilot_d/rank_mobility.csv` has every rep, lag and measure)

α = 0 is the standalone baseline (every firm a standalone).

**Within markets, h = 100** (mean over the 3 reps; all firms / hit-free)

| family | α | Spearman | top-decile persistence | bottom-decile persistence | leader turnover | mean abs rank change | share with a lift in (t, t+h] | firms per hit-free market |
|---|---|---|---|---|---|---|---|---|
| normal | 0 | 0.329 / 0.411 | 0.476 / 0.453 | 0.129 / 0.173 | 0.598 / 0.623 | 13.09 / 4.80 | 0.677 | 18.5 |
| normal | 0.05 | 0.343 / 0.419 | 0.483 / 0.457 | 0.130 / 0.173 | 0.593 / 0.616 | 12.93 / 4.93 | 0.658 | 19.1 |
| normal | 0.1 | 0.362 / 0.430 | 0.490 / 0.468 | 0.134 / 0.177 | 0.593 / 0.616 | 12.71 / 5.08 | 0.634 | 19.9 |
| normal | 0.3 | 0.454 / 0.494 | 0.530 / 0.499 | 0.156 / 0.190 | 0.582 / 0.587 | 11.61 / 5.73 | 0.521 | 24.5 |
| Laplace | 0 | 0.214 / 0.310 | 0.353 / 0.358 | 0.118 / 0.155 | 0.730 / 0.712 | 14.39 / 4.44 | 0.783 | 15.8 |
| Laplace | 0.05 | 0.225 / 0.312 | 0.359 / 0.353 | 0.118 / 0.149 | 0.728 / 0.712 | 14.27 / 4.56 | 0.766 | 16.2 |
| Laplace | 0.1 | 0.239 / 0.317 | 0.366 / 0.348 | 0.121 / 0.158 | 0.723 / 0.717 | 14.11 / 4.67 | 0.746 | 16.6 |
| Laplace | 0.3 | 0.316 / 0.355 | 0.408 / 0.373 | 0.135 / 0.166 | 0.711 / 0.707 | 13.25 / 5.28 | 0.644 | 19.4 |
| t₃ | 0 | 0.164 / 0.293 | 0.310 / 0.316 | 0.113 / 0.148 | 0.769 / 0.739 | 14.92 / 4.31 | 0.826 | 15.1 |
| t₃ | 0.05 | 0.173 / 0.290 | 0.316 / 0.320 | 0.115 / 0.155 | 0.771 / 0.740 | 14.82 / 4.44 | 0.812 | 15.4 |
| t₃ | 0.1 | 0.185 / 0.293 | 0.322 / 0.323 | 0.116 / 0.138 | 0.767 / 0.733 | 14.69 / 4.55 | 0.795 | 15.8 |
| t₃ | 0.3 | 0.254 / 0.325 | 0.363 / 0.356 | 0.128 / 0.153 | 0.752 / 0.718 | 13.93 / 4.96 | 0.711 | 17.7 |

**By status at t, h = 100** (members / standalones; all firms, then hit-free)

| family | α | top-decile persistence | bottom-decile persistence | mean abs rank change | members' share of the top decile | lift in (t, t+h] |
|---|---|---|---|---|---|---|
| normal | 0.05 | 0.458 / 0.646; 0.418 / 0.590 | 0.126 / 0.130; 0.172 / 0.174 | 12.25 / 14.02; 4.87 / 5.17 | 0.867 | 0.552 / 0.829 |
| normal | 0.1 | 0.480 / 0.634; 0.450 / 0.634 | 0.135 / 0.134; 0.183 / 0.169 | 11.98 / 14.07; 4.97 / 5.65 | 0.938 | 0.522 / 0.842 |
| normal | 0.3 | 0.530 / 0.483; 0.499 / 0.494 | 0.167 / 0.153; 0.200 / 0.173 | 10.88 / 13.63; 5.59 / 7.08 | 0.988 | 0.412 / 0.821 |
| Laplace | 0.05 | 0.349 / 0.425; 0.324 / 0.474 | 0.127 / 0.118; 0.149 / 0.149 | 13.63 / 15.06; 4.50 / 4.78 | 0.876 | 0.671 / 0.885 |
| Laplace | 0.1 | 0.366 / 0.364; 0.339 / 0.448 | 0.143 / 0.119; 0.155 / 0.160 | 13.41 / 15.11; 4.59 / 5.05 | 0.928 | 0.646 / 0.888 |
| Laplace | 0.3 | 0.411 / 0.298; 0.375 / 0.282 | 0.151 / 0.132; 0.167 / 0.163 | 12.47 / 14.85; 5.15 / 6.23 | 0.972 | 0.536 / 0.867 |
| t₃ | 0.05 | 0.311 / 0.342; 0.309 / 0.362 | 0.112 / 0.115; 0.154 / 0.156 | 14.17 / 15.52; 4.35 / 4.68 | 0.842 | 0.726 / 0.904 |
| t₃ | 0.1 | 0.326 / 0.291; 0.314 / 0.412 | 0.107 / 0.117; 0.138 / 0.137 | 13.95 / 15.56; 4.45 / 4.90 | 0.895 | 0.702 / 0.907 |
| t₃ | 0.3 | 0.368 / 0.276; 0.358 / 0.274 | 0.142 / 0.127; 0.162 / 0.140 | 13.13 / 15.28; 4.83 / 5.75 | 0.953 | 0.603 / 0.892 |

**Owners across the economy** (a unit counted once; Spearman, top-decile persistence, leader turnover at each lag; top-decile capital share)

| family | α | h = 10: ρ / top / leader | h = 50: ρ / top / leader | h = 100: ρ / top / leader | h = 500: ρ / top / leader | top-decile share (trace) | top10pct_aggregate (pickle) | owners |
|---|---|---|---|---|---|---|---|---|
| normal | 0 | 0.806 / 0.824 / 0.57 | 0.495 / 0.625 / 0.78 | 0.325 / 0.490 / 0.89 | 0.045 / 0.173 / 1.00 | 0.652 | 0.654 | 2500 |
| normal | 0.05 | 0.771 / 0.870 / 0.37 | 0.532 / 0.733 / 0.68 | 0.376 / 0.634 / 0.84 | 0.059 / 0.291 / 0.96 | 0.666 | 0.666 | 1449 |
| normal | 0.1 | 0.753 / 0.878 / 0.43 | 0.524 / 0.743 / 0.76 | 0.381 / 0.659 / 0.89 | 0.070 / 0.336 / 1.00 | 0.675 | 0.677 | 1317 |
| normal | 0.3 | 0.740 / 0.896 / 0.38 | 0.523 / 0.764 / 0.68 | 0.403 / 0.686 / 0.79 | 0.101 / 0.424 / 0.99 | 0.673 | 0.672 | 998 |
| Laplace | 0 | 0.725 / 0.771 / 0.70 | 0.361 / 0.516 / 0.97 | 0.205 / 0.365 / 0.99 | 0.020 / 0.126 / 1.00 | 0.654 | 0.654 | 2500 |
| Laplace | 0.05 | 0.685 / 0.843 / 0.59 | 0.390 / 0.665 / 0.83 | 0.238 / 0.552 / 0.91 | 0.022 / 0.260 / 1.00 | 0.696 | 0.696 | 1548 |
| Laplace | 0.1 | 0.672 / 0.860 / 0.49 | 0.391 / 0.703 / 0.80 | 0.239 / 0.589 / 0.91 | 0.025 / 0.298 / 1.00 | 0.712 | 0.713 | 1433 |
| Laplace | 0.3 | 0.658 / 0.884 / 0.55 | 0.399 / 0.755 / 0.79 | 0.275 / 0.677 / 0.94 | 0.032 / 0.406 / 1.00 | 0.731 | 0.732 | 1131 |
| t₃ | 0 | 0.687 / 0.764 / 0.75 | 0.309 / 0.482 / 0.98 | 0.156 / 0.323 / 0.98 | 0.008 / 0.120 / 1.00 | 0.681 | 0.679 | 2500 |
| t₃ | 0.05 | 0.645 / 0.836 / 0.60 | 0.335 / 0.644 / 0.87 | 0.185 / 0.502 / 0.95 | 0.008 / 0.240 / 1.00 | 0.720 | 0.718 | 1626 |
| t₃ | 0.1 | 0.636 / 0.852 / 0.56 | 0.335 / 0.684 / 0.83 | 0.192 / 0.552 / 0.98 | 0.013 / 0.279 / 0.99 | 0.736 | 0.733 | 1530 |
| t₃ | 0.3 | 0.621 / 0.884 / 0.46 | 0.352 / 0.758 / 0.77 | 0.226 / 0.672 / 0.87 | 0.019 / 0.382 / 1.00 | 0.767 | 0.768 | 1259 |

**Spearman by lag** (all firms / hit-free)

| family | α | h = 10 | h = 50 | h = 100 | h = 500 |
|---|---|---|---|---|---|
| normal | 0 | 0.794 / 0.816 | 0.494 / 0.564 | 0.329 / 0.411 | 0.047 / 0.213 |
| normal | 0.05 | 0.804 / 0.823 | 0.510 / 0.573 | 0.343 / 0.419 | 0.051 / 0.183 |
| normal | 0.1 | 0.816 / 0.831 | 0.529 / 0.583 | 0.362 / 0.430 | 0.055 / 0.200 |
| normal | 0.3 | 0.867 / 0.872 | 0.619 / 0.647 | 0.454 / 0.494 | 0.088 / 0.179 |
| Laplace | 0 | 0.716 / 0.751 | 0.366 / 0.444 | 0.214 / 0.310 | 0.021 / 0.152 |
| Laplace | 0.05 | 0.728 / 0.758 | 0.381 / 0.451 | 0.225 / 0.312 | 0.022 / 0.103 |
| Laplace | 0.1 | 0.742 / 0.769 | 0.399 / 0.461 | 0.239 / 0.317 | 0.023 / 0.079 |
| Laplace | 0.3 | 0.803 / 0.816 | 0.490 / 0.524 | 0.316 / 0.355 | 0.036 / 0.140 |
| t₃ | 0 | 0.682 / 0.735 | 0.320 / 0.438 | 0.164 / 0.293 | 0.009 / 0.108 |
| t₃ | 0.05 | 0.693 / 0.741 | 0.332 / 0.438 | 0.173 / 0.290 | 0.009 / 0.049 |
| t₃ | 0.1 | 0.708 / 0.750 | 0.350 / 0.447 | 0.185 / 0.293 | 0.011 / 0.162 |
| t₃ | 0.3 | 0.766 / 0.792 | 0.431 / 0.495 | 0.254 / 0.325 | 0.018 / 0.169 |

**Per-firm rank range and rank SD, post-burn-in (3,000 periods)** (distribution over the 2,500 firms; mean over the 3 reps)

| family | α | range p10 / p25 / median / p75 | share at max (49) | SD p10 / p25 / median / p75 / p90 | SD mean |
|---|---|---|---|---|---|
| normal | 0 | 46.7 / 48.0 / 49.0 / 49.0 | 0.549 | 12.25 / 13.02 / 13.87 / 14.66 / 15.34 | 13.81 |
| normal | 0.05 | 46.3 / 48.0 / 49.0 / 49.0 | 0.548 | 12.20 / 12.99 / 13.84 / 14.65 / 15.33 | 13.79 |
| normal | 0.1 | 46.0 / 47.7 / 49.0 / 49.0 | 0.541 | 12.11 / 12.94 / 13.81 / 14.63 / 15.32 | 13.75 |
| normal | 0.3 | 45.7 / 47.0 / 48.3 / 49.0 | 0.507 | 11.71 / 12.65 / 13.62 / 14.52 / 15.34 | 13.54 |
| Laplace | 0 | 47.7 / 48.7 / 49.0 / 49.0 | 0.735 | 12.69 / 13.35 / 14.08 / 14.75 / 15.37 | 14.04 |
| Laplace | 0.05 | 47.7 / 48.7 / 49.0 / 49.0 | 0.733 | 12.67 / 13.34 / 14.07 / 14.74 / 15.36 | 14.02 |
| Laplace | 0.1 | 47.7 / 48.3 / 49.0 / 49.0 | 0.731 | 12.63 / 13.31 / 14.04 / 14.73 / 15.35 | 14.00 |
| Laplace | 0.3 | 47.3 / 48.0 / 49.0 / 49.0 | 0.696 | 12.36 / 13.15 / 13.93 / 14.68 / 15.31 | 13.87 |
| t₃ | 0 | 48.0 / 48.7 / 49.0 / 49.0 | 0.748 | 12.83 / 13.42 / 14.13 / 14.81 / 15.39 | 14.11 |
| t₃ | 0.05 | 48.0 / 48.7 / 49.0 / 49.0 | 0.749 | 12.80 / 13.42 / 14.13 / 14.80 / 15.38 | 14.10 |
| t₃ | 0.1 | 48.0 / 48.3 / 49.0 / 49.0 | 0.744 | 12.77 / 13.40 / 14.12 / 14.81 / 15.37 | 14.08 |
| t₃ | 0.3 | 47.0 / 48.0 / 49.0 / 49.0 | 0.723 | 12.59 / 13.26 / 14.05 / 14.74 / 15.34 | 13.99 |

**Per-firm rank range and rank SD, whole run (11,000 periods, as the model)** (distribution over the 2,500 firms; mean over the 3 reps)

| family | α | range p10 / p25 / median / p75 | share at max (49) | SD p10 / p25 / median / p75 / p90 | SD mean |
|---|---|---|---|---|---|
| normal | 0 | 49.0 / 49.0 / 49.0 / 49.0 | 0.931 | 13.29 / 13.78 / 14.27 / 14.74 / 15.16 | 14.25 |
| normal | 0.05 | 49.0 / 49.0 / 49.0 / 49.0 | 0.929 | 13.28 / 13.77 / 14.26 / 14.73 / 15.15 | 14.24 |
| normal | 0.1 | 49.0 / 49.0 / 49.0 / 49.0 | 0.924 | 13.26 / 13.76 / 14.25 / 14.74 / 15.15 | 14.23 |
| normal | 0.3 | 49.0 / 49.0 / 49.0 / 49.0 | 0.908 | 13.13 / 13.67 / 14.19 / 14.72 / 15.16 | 14.17 |
| Laplace | 0 | 49.0 / 49.0 / 49.0 / 49.0 | 0.980 | 13.48 / 13.86 / 14.31 / 14.77 / 15.17 | 14.32 |
| Laplace | 0.05 | 49.0 / 49.0 / 49.0 / 49.0 | 0.980 | 13.48 / 13.87 / 14.31 / 14.76 / 15.16 | 14.31 |
| Laplace | 0.1 | 49.0 / 49.0 / 49.0 / 49.0 | 0.978 | 13.48 / 13.87 / 14.31 / 14.75 / 15.14 | 14.31 |
| Laplace | 0.3 | 49.0 / 49.0 / 49.0 / 49.0 | 0.971 | 13.43 / 13.84 / 14.29 / 14.72 / 15.11 | 14.28 |
| t₃ | 0 | 49.0 / 49.0 / 49.0 / 49.0 | 0.985 | 13.50 / 13.86 / 14.31 / 14.79 / 15.21 | 14.33 |
| t₃ | 0.05 | 49.0 / 49.0 / 49.0 / 49.0 | 0.985 | 13.51 / 13.87 / 14.31 / 14.77 / 15.20 | 14.33 |
| t₃ | 0.1 | 49.0 / 49.0 / 49.0 / 49.0 | 0.983 | 13.52 / 13.87 / 14.31 / 14.77 / 15.18 | 14.33 |
| t₃ | 0.3 | 49.0 / 49.0 / 49.0 / 49.0 | 0.982 | 13.48 / 13.87 / 14.30 / 14.74 / 15.12 | 14.30 |

At h = 500 the lift-free sample has 11–14 firms per market and many markets have fewer than 10, so its h = 500
values (in the CSV) are noisy, e.g. t₃ α = 0 bottom-decile persistence 0.000 on few markets.
