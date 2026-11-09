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

## 2. `decision_rule='accept'`: the survival-feasible ceiling

**Caveat (Simon's).** The accept rule is a control, not a behavioural alternative. It has no knowledge at all. The
comparison measures what the knowledge in the moment rule is worth, not whether firms "should" accept more.

**Rule.** `decision_rule='accept'` (code 4) accepts a proposal if and only if it is feasible: the merged set holds at
most one firm per market.
- No member gain is computed, and the exit review never produces a voluntary exit.
- Floor exits, limited liability, pooling, cost and the floor are unchanged. α is the scenario's fixed rate.
- The acceptance counters stay on, so the accepted proposals' composition is recorded. The binding-gain sums stay
  empty.
- As under every rule, no proposal is made at step 0 (window h = 0).
- MODEL_FLAGS.md has the entry.

**Tests** (`tests/test_accept_rule.py`):
- (a) On a hand-built three-market economy, every proposal reaching the test merges, while the moment rule refuses
  some. On an evolving five-market economy, no unit ever holds two firms of one market, and every period's
  mergers equal its feasible proposals. A brute-force feasibility oracle agrees on which pairs can merge.
- (b) The exit kernel returns no firm under accept for a member that leaves under moment. In a model run every exit
  from a unit is a floor exit.
- (c) The identity check below.
- (d) The pooling identity Σ s_j(1 + r_j) − Φ(K)S holds to 1e-9 S in every unit-period of accept runs under both
  sharing rules.

**Identity.** All 2,055 existing pilot D scenarios (C37, C38 α = 0.9, C39 cv) were rerun on the C42 code and compared
with their C39 pickles, which carry every C38 and C39 key (`analytics/identity_c42.py`,
`diagnostics/identity_c42.csv`). 2,055 of 2,055 are byte-identical in every field and in the tidy row, with no new
key. The median runtime ratio is 0.99.

**Runs.** The `accept` block holds the main power-law cells, three families, α ∈ {0.1, 0.3, 0.5}, 5 reps, on the
main seeds: ids 2055–2099, results in `pilot_d/results_c42/`. The rows are appended to `scenarios.json` and
`tidy.csv`; the existing 2,055 entries are byte-identical. Each run takes about 2 minutes (moment: about 5).

**Answer.**

**(i) How far above K_moment is the ceiling?** Three to five times as high.
- K_accept is 12.4–28.2 members against K_moment's 3.5–7.7, a ratio of 3.1–4.8.
- Paired differences are +8.9 to +20.5 members, with tight IQRs.
- Consent therefore costs most of the scope the environment permits: 68–79% of K_accept is never formed.
- Units hold 8.5–10 percentage points more of the firms, and the pooled capital share is 0.02–0.11 higher.
- Concentration barely moves:
  - Hill rises by 0.00 to +0.04;
  - within-market HHI falls slightly, by 0.004–0.012.
- The extra mergers per period, +2.8 to +13.4, equal the extra floor exits. In the steady state recruits flow in
  and members are ejected at the floor at the same rate.

**(ii) Unit+unit mergers or small recruits?** In flows, small recruits.
- At α = 0.3, accept merges 6.5–8.6 more proposals per period than moment.
- All of the excess is solo+unit entries (+12.5 to +17.0), mostly:
  - floor-sized standalones, +7.6 to +8.8;
  - entrants with w ≤ 0.10, +1.8 to +3.5.
- Solo+solo mergers fall (−4.8 to −7.7), and so do unit+unit mergers (−0.8 to −1.2).
- The moment rule refuses as many unit+unit proposals as solo+unit ones: 23–30 against 31–33 per period at
  α = 0.3. Yet unit+unit mergers do not rise once consent is off.
  - With every feasible proposal accepted, the economy consolidates into fewer, larger units.
  - Units then occupy most markets, and most unit pairs cannot merge without a market clash.
  - The unit+unit mergers that remain are mostly a tiny unit joining a giant: entrant w ≤ 0.10 in 82–92% of them,
    against 18–26% under moment.
- So, once formed, the large units are kept large by a stream of small recruits that balances floor ejection.
- The flows cannot tell how the large units were first assembled, by unit+unit mergers or one recruit at a time,
  so they cannot split the stock K between the two channels. A run that accepts only one type of proposal would;
  it was not run.

**(iii) Does attrition still keep the accept economy well below K\*?** Only for t₃.
- In normal and Laplace, K_accept exceeds K\* in all six cells:
  - normal: 17.6 / 22.2 / 28.2 against K\* = 8 / 12 / 13, about 1.9–2.2 times K\*;
  - Laplace: 13.8 / 18.1 / 24.3 against 10 / 16 / 20, about 1.1–1.4 times.
- There the environment does not keep K below the planner's scope. Attrition and the one-firm-per-market
  constraint would allow units larger than K\*. The whole distance K\* − K_hom, the first column of Table 3c and
  75–89% of its total, is consent.
- Under accept, members' floor-hit rates in these cells barely move: −0.0001 to +0.0031 per member-period.
- In t₃, accept stays well below K\*: 12.4 / 16.2 / 21.8 against 20 / 47 / 50, about 0.3–0.6 times K\*.
  - There attrition binds: members hit the floor at 0.018–0.043 per period, the highest rates of the three
    families.
  - K\* = 47–50 is also at the one-firm-per-market limit (M = 50).
  - The environment accounts for 7.6 / 30.8 / 28.2 members (44–71% of the total) and consent for 8.9–14.8.

**Table 3c re-cut** (`pilot_d/consent_gap_decomposition_accept.csv`, `latex/tables/table3c_decomposition_accept.tex`
from `analysis/paper_tables.py --table3c-accept`; the existing `table3c_decomposition.tex` is unchanged):
- environment = K\* − K_accept;
- consent on recruit size and noise = K_accept − K_hom;
- transfer conflict = K_hom − K;
- verifiability = K − K_loggain.

The first two sum to Table 3c's first part.
- **Negative environment parts.** A negative environment part (normal, Laplace) means the environment permits more
  than K\*. The consent part then exceeds 100% of the total.
- **Not a clean split.** K_accept is the heterogeneous economy and K_hom the homogeneous one, so the consent part
  also holds the homogeneous–heterogeneous difference of the accept rule. That difference was not run.

| family | α | K* | K_accept | K_hom | K | K_loggain | environment K* − K_accept | consent (size, noise) K_accept − K_hom | transfer K_hom − K | verifiability K − K_loggain |
|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.1 | 8 | 17.60 | 3.78 | 3.68 | 3.04 | -9.6 (-194%) | 13.8 (279%) | 0.1 (2%) | 0.6 (13%) |
| normal | 0.3 | 12 | 22.24 | 5.57 | 5.48 | 4.29 | -10.2 (-133%) | 16.7 (216%) | 0.1 (1%) | 1.2 (15%) |
| normal | 0.5 | 13 | 28.22 | 7.98 | 7.70 | 6.34 | -15.2 (-228%) | 20.2 (304%) | 0.3 (4%) | 1.4 (20%) |
| laplace | 0.1 | 10 | 13.82 | 3.81 | 3.71 | 3.00 | -3.8 (-55%) | 10.0 (143%) | 0.1 (1%) | 0.7 (10%) |
| laplace | 0.3 | 16 | 18.10 | 5.55 | 5.40 | 4.00 | -2.1 (-18%) | 12.6 (105%) | 0.1 (1%) | 1.4 (12%) |
| laplace | 0.5 | 20 | 24.30 | 8.17 | 7.76 | 5.95 | -4.3 (-31%) | 16.1 (115%) | 0.4 (3%) | 1.8 (13%) |
| t3 | 0.1 | 20 | 12.41 | 3.52 | 3.47 | 2.87 | 7.6 (44%) | 8.9 (52%) | 0.0 (0%) | 0.6 (4%) |
| t3 | 0.3 | 47 | 16.23 | 4.92 | 4.84 | 3.67 | 30.8 (71%) | 11.3 (26%) | 0.1 (0%) | 1.2 (3%) |
| t3 | 0.5 | 50 | 21.77 | 6.96 | 6.78 | 5.06 | 28.2 (63%) | 14.8 (33%) | 0.2 (0%) | 1.7 (4%) |

### Accept / moment / loggain (medians over the 5 reps; `pilot_d/accept_comparison.csv`)

| family | α | K accept / moment / loggain | pooled share accept / moment / loggain | Hill accept / moment / loggain | HHI within accept / moment / loggain | floor exits / period accept / moment / loggain | mergers / period accept / moment / loggain | voluntary exits / period accept / moment / loggain |
|---|---|---|---|---|---|---|---|---|
| normal | 0.1 | 17.60 / 3.68 / 3.04 | 0.954 / 0.850 / 0.561 | 1.124 / 1.113 / 1.087 | 0.1601 / 0.1642 / 0.1689 | 46.0 / 34.1 / 34.8 | 46.0 / 34.2 / 37.7 | 0.00 / 0.06 / 2.99 |
| normal | 0.3 | 22.24 / 5.48 / 4.29 | 0.967 / 0.939 / 0.733 | 1.207 / 1.209 / 1.169 | 0.1334 / 0.1385 / 0.1477 | 31.8 / 25.0 / 26.0 | 31.8 / 25.0 / 27.3 | 0.00 / 0.00 / 1.37 |
| normal | 0.5 | 28.22 / 7.70 / 6.34 | 0.980 / 0.956 / 0.912 | 1.373 / 1.342 / 1.316 | 0.1064 / 0.1085 / 0.1182 | 18.4 / 15.6 / 16.1 | 18.4 / 15.6 / 16.3 | 0.00 / 0.00 / 0.24 |
| laplace | 0.1 | 13.82 / 3.71 / 3.00 | 0.927 / 0.853 / 0.634 | 1.110 / 1.105 / 1.098 | 0.1636 / 0.1676 / 0.1735 | 59.7 / 47.4 / 46.8 | 59.8 / 47.4 / 48.6 | 0.00 / 0.02 / 1.78 |
| laplace | 0.3 | 18.10 / 5.40 / 4.00 | 0.948 / 0.914 / 0.782 | 1.230 / 1.195 / 1.165 | 0.1353 / 0.1439 / 0.1509 | 43.9 / 36.3 / 36.9 | 43.9 / 36.3 / 37.8 | 0.00 / 0.00 / 0.85 |
| laplace | 0.5 | 24.30 / 7.76 / 5.95 | 0.966 / 0.934 / 0.906 | 1.347 / 1.344 / 1.329 | 0.1085 / 0.1133 / 0.1205 | 27.5 / 24.5 / 25.1 | 27.5 / 24.5 / 25.2 | 0.00 / 0.00 / 0.18 |
| t3 | 0.1 | 12.41 / 3.47 / 2.87 | 0.915 / 0.827 / 0.627 | 1.041 / 1.035 / 1.018 | 0.1882 / 0.1914 / 0.1963 | 66.4 / 53.1 / 52.1 | 66.5 / 53.1 / 53.7 | 0.00 / 0.04 / 1.59 |
| t3 | 0.3 | 16.23 / 4.84 / 3.67 | 0.937 / 0.896 / 0.773 | 1.130 / 1.116 / 1.096 | 0.1597 / 0.1665 / 0.1744 | 51.1 / 42.5 / 43.2 | 51.1 / 42.5 / 44.1 | 0.00 / 0.00 / 0.93 |
| t3 | 0.5 | 21.77 / 6.78 / 5.06 | 0.957 / 0.920 / 0.884 | 1.259 / 1.221 / 1.182 | 0.1273 / 0.1357 / 0.1445 | 34.7 / 31.2 / 32.3 | 34.7 / 31.2 / 32.6 | 0.00 / 0.00 / 0.31 |

Paired differences accept − moment (median [p25, p75] over the 5 reps)

| family | α | K | pooled share | Hill | HHI within | floor exits / period | mergers / period | member floor-hit rate | share of firms in units |
|---|---|---|---|---|---|---|---|---|---|
| normal | 0.1 | +13.86 [+13.50, +14.17] | +0.106 [+0.103, +0.106] | +0.025 [+0.020, +0.030] | -0.0042 [-0.0053, -0.0041] | +11.4 [+11.3, +11.5] | +11.3 [+11.2, +11.5] | +0.0031 [+0.0031, +0.0032] | +0.101 [+0.099, +0.102] |
| normal | 0.3 | +16.70 [+16.69, +17.02] | +0.028 [+0.027, +0.028] | -0.001 [-0.004, +0.016] | -0.0059 [-0.0073, -0.0051] | +6.4 [+6.3, +6.6] | +6.4 [+6.3, +6.6] | +0.0016 [+0.0015, +0.0017] | +0.088 [+0.088, +0.090] |
| normal | 0.5 | +20.48 [+20.43, +20.78] | +0.024 [+0.023, +0.024] | +0.029 [+0.019, +0.050] | -0.0047 [-0.0065, -0.0021] | +2.8 [+2.8, +2.8] | +2.8 [+2.8, +2.8] | +0.0004 [+0.0004, +0.0005] | +0.085 [+0.081, +0.087] |
| laplace | 0.1 | +10.10 [+9.35, +10.98] | +0.072 [+0.070, +0.085] | +0.017 [+0.002, +0.020] | -0.0061 [-0.0063, -0.0057] | +12.4 [+12.3, +12.7] | +12.3 [+12.3, +12.7] | +0.0026 [+0.0024, +0.0026] | +0.094 [+0.091, +0.095] |
| laplace | 0.3 | +12.69 [+11.58, +13.38] | +0.034 [+0.033, +0.034] | +0.027 [+0.014, +0.032] | -0.0118 [-0.0122, -0.0086] | +7.6 [+7.3, +8.1] | +7.6 [+7.3, +8.1] | +0.0013 [+0.0012, +0.0013] | +0.089 [+0.087, +0.089] |
| laplace | 0.5 | +16.57 [+15.86, +17.62] | +0.031 [+0.030, +0.032] | +0.010 [+0.010, +0.025] | -0.0046 [-0.0075, -0.0044] | +3.3 [+3.2, +3.5] | +3.2 [+3.2, +3.5] | -0.0001 [-0.0001, +0.0002] | +0.089 [+0.088, +0.091] |
| t3 | 0.1 | +8.94 [+8.85, +8.97] | +0.085 [+0.085, +0.090] | +0.017 [+0.007, +0.017] | -0.0039 [-0.0058, -0.0033] | +13.4 [+13.4, +13.4] | +13.4 [+13.4, +13.4] | +0.0020 [+0.0020, +0.0021] | +0.096 [+0.095, +0.097] |
| t3 | 0.3 | +11.41 [+11.13, +11.44] | +0.041 [+0.041, +0.042] | +0.011 [+0.001, +0.025] | -0.0073 [-0.0094, -0.0067] | +8.5 [+8.4, +8.7] | +8.5 [+8.4, +8.7] | +0.0007 [+0.0007, +0.0008] | +0.095 [+0.095, +0.096] |
| t3 | 0.5 | +15.08 [+14.87, +15.10] | +0.037 [+0.035, +0.037] | +0.040 [+0.032, +0.059] | -0.0085 [-0.0097, -0.0079] | +3.8 [+3.6, +4.1] | +3.8 [+3.6, +4.0] | -0.0008 [-0.0010, -0.0005] | +0.102 [+0.102, +0.105] |

Member floor-hit rate (per member-period) and share of firms in units, accept / moment

| family | α | member floor-hit rate | share of firms in units |
|---|---|---|---|
| normal | 0.1 | 0.0248 / 0.0214 | 0.751 / 0.652 |
| normal | 0.3 | 0.0156 / 0.0139 | 0.825 / 0.736 |
| normal | 0.5 | 0.0083 / 0.0079 | 0.894 / 0.808 |
| laplace | 0.1 | 0.0363 / 0.0337 | 0.671 / 0.580 |
| laplace | 0.3 | 0.0236 / 0.0223 | 0.757 / 0.669 |
| laplace | 0.5 | 0.0133 / 0.0135 | 0.840 / 0.747 |
| t3 | 0.1 | 0.0427 / 0.0406 | 0.640 / 0.543 |
| t3 | 0.3 | 0.0288 / 0.0283 | 0.721 / 0.626 |
| t3 | 0.5 | 0.0176 / 0.0186 | 0.803 / 0.700 |

### Acceptance and mergers by K, α = 0.3 (pooled over reps; every α in `pilot_d/accept_acceptance_by_K.csv`)

Under accept, acceptance is 1 at every K by construction, so the table shows mergers per period by K for all three
rules and acceptance for moment and loggain.

| K | normal mergers | laplace mergers | t3 mergers | normal acceptance | laplace acceptance | t3 acceptance |
|---|---|---|---|---|---|---|
| 2 | 3.50 / 8.28 / 10.58 | 6.85 / 13.06 / 15.88 | 9.06 / 16.76 / 19.94 | 1.00 / 0.92 | 1.00 / 0.96 | 1.00 / 0.97 |
| 3 | 0.49 / 2.71 / 4.09 | 1.11 / 4.08 / 5.88 | 1.60 / 5.32 / 7.42 | 0.90 / 0.55 | 0.87 / 0.59 | 0.76 / 0.57 |
| 4 | 0.15 / 1.89 / 2.85 | 0.38 / 2.60 / 3.87 | 0.58 / 3.36 / 4.73 | 0.68 / 0.37 | 0.68 / 0.41 | 0.59 / 0.39 |
| 5 | 0.07 / 1.92 / 2.34 | 0.20 / 2.39 / 3.05 | 0.31 / 2.95 / 3.49 | 0.47 / 0.28 | 0.54 / 0.31 | 0.48 / 0.30 |
| 6 | 0.05 / 2.19 / 1.98 | 0.14 / 2.51 / 2.47 | 0.22 / 2.82 / 2.67 | 0.35 / 0.21 | 0.43 / 0.25 | 0.40 / 0.24 |
| 7 | 0.04 / 2.26 / 1.64 | 0.12 / 2.59 / 1.96 | 0.18 / 2.71 / 1.94 | 0.26 / 0.17 | 0.35 / 0.20 | 0.34 / 0.20 |
| 8 | 0.03 / 1.96 / 1.26 | 0.11 / 2.51 / 1.53 | 0.17 / 2.42 / 1.37 | 0.20 / 0.13 | 0.30 / 0.17 | 0.28 / 0.16 |
| 9 | 0.04 / 1.46 / 0.87 | 0.12 / 2.12 / 1.08 | 0.18 / 2.02 / 0.93 | 0.16 / 0.10 | 0.24 / 0.14 | 0.24 / 0.14 |
| 10 | 0.04 / 0.94 / 0.54 | 0.13 / 1.60 / 0.71 | 0.21 / 1.50 / 0.60 | 0.12 / 0.07 | 0.20 / 0.11 | 0.20 / 0.11 |
| 11 | 0.06 / 0.50 / 0.30 | 0.17 / 1.11 / 0.43 | 0.26 / 1.02 / 0.35 | 0.07 / 0.05 | 0.16 / 0.09 | 0.16 / 0.09 |
| 12 | 0.08 / 0.23 / 0.16 | 0.20 / 0.67 / 0.25 | 0.32 / 0.64 / 0.19 | 0.04 / 0.04 | 0.12 / 0.07 | 0.13 / 0.07 |
| 13 | 0.10 / 0.09 / 0.08 | 0.27 / 0.37 / 0.13 | 0.39 / 0.36 / 0.10 | 0.02 / 0.02 | 0.09 / 0.05 | 0.10 / 0.05 |
| 14 | 0.13 / 0.03 / 0.04 | 0.35 / 0.18 / 0.07 | 0.51 / 0.20 / 0.05 | 0.01 / 0.02 | 0.06 / 0.04 | 0.07 / 0.04 |
| 15 | 0.17 / 0.01 / 0.01 | 0.44 / 0.09 / 0.03 | 0.62 / 0.10 / 0.02 | 0.00 / 0.01 | 0.04 / 0.03 | 0.05 / 0.03 |
| 16 | 0.22 / 0.00 / 0.01 | 0.54 / 0.03 / 0.01 | 0.76 / 0.04 / 0.01 | 0.00 / 0.01 | 0.02 / 0.02 | 0.03 / 0.02 |
| 17 | 0.30 / 0.00 / 0.00 | 0.66 / 0.01 / 0.01 | 0.92 / 0.02 / 0.00 | 0.00 / 0.00 | 0.01 / 0.02 | 0.02 / 0.02 |
| 18 | 0.37 / 0.00 / 0.00 | 0.79 / 0.00 / 0.00 | 1.08 / 0.01 / 0.00 | 0.00 / 0.00 | 0.01 / 0.01 | 0.01 / 0.01 |
| 19 | 0.46 / 0.00 / 0.00 | 0.95 / 0.00 / 0.00 | 1.26 / 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.01 | 0.01 / 0.01 |
| 20 | 0.56 / 0.00 / 0.00 | 1.09 / 0.00 / 0.00 | 1.47 / 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

### Composition of the accepted proposals, α = 0.3 (accept against moment; every α in `pilot_d/accept_composition.csv`)

**α = 0.3: accepted proposals by type**, per period (share of all mergers): accept / moment; moment refusals per period

| family | solo+solo | solo+unit | unit+unit | refused by moment: solo+solo / solo+unit / unit+unit |
|---|---|---|---|---|
| normal | 3.50 (11%) / 8.28 (34%) | 26.54 (86%) / 14.08 (57%) | 0.94 (3%) / 2.13 (9%) | 0.01 / 31.23 / 30.40 |
| laplace | 6.85 (16%) / 13.06 (36%) | 35.00 (80%) / 19.92 (55%) | 1.70 (4%) / 2.94 (8%) | 0.00 / 31.54 / 23.12 |
| t3 | 9.06 (18%) / 16.76 (40%) | 39.55 (78%) / 22.54 (53%) | 2.18 (4%) / 2.94 (7%) | 0.00 / 32.57 / 22.81 |

**α = 0.3: entrant class of the accepted solo entries** (solo+solo and solo+unit; share of those mergers) accept / moment, and unit+unit by entrant share

| family | floor | w<=0.10 | 0.10-0.25 | 0.25-0.50 | w>0.50 | unit+unit: w<=0.10 / 0.10-0.25 / 0.25-0.50 / w>0.50 |
|---|---|---|---|---|---|---|
| normal | 89% / 86% | 10% / 6% | 0% / 4% | 0% / 4% | 0% / 1% | 92% / 4% / 4% / 0% vs 18% / 36% / 46% / 0% |
| laplace | 86% / 83% | 13% / 7% | 0% / 4% | 1% / 5% | 0% / 1% | 85% / 8% / 6% / 0% vs 26% / 33% / 41% / 0% |
| t3 | 86% / 85% | 12% / 6% | 1% / 4% | 1% / 5% | 0% / 1% | 82% / 10% / 8% / 0% vs 24% / 32% / 44% / 0% |

**α = 0.3: excess mergers per period, accept − moment**, by type and (solo entries) entrant class

| family | total | solo+solo | solo+unit | unit+unit | solo entries: floor | w ≤ 0.10 | 0.10–0.25 | > 0.25 |
|---|---|---|---|---|---|---|---|---|
| normal | +6.48 | -4.79 | +12.46 | -1.19 | +7.63 | +1.77 | -0.84 | -0.90 |
| laplace | +7.62 | -6.22 | +15.08 | -1.24 | +8.35 | +3.11 | -1.12 | -1.47 |
| t3 | +8.55 | -7.70 | +17.01 | -0.76 | +8.75 | +3.48 | -1.18 | -1.74 |

