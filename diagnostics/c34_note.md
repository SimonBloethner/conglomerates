# C34 note: limited liability, demeaned renegotiation, notation rename

Commits on `referee-fixes`: C34a `7d5018f` (limited liability), C34b `cc47254` (demeaned renegotiation), C34c `b020237` (rename), then one commit with the reruns and regenerated outputs. Tests on Festus: 206 passed after C34a, 207 after C34b, 218 after C34c and on the final state. `collaborative_growth.py` at C34c is the new frozen model, and every pilot number now comes from C34.

## Conservation test

For every active unit in every period, |Σ_j s_j,t+1 − [Σ_j s_j(1 + r_j) − Φ(K)·S]| / S:
- Seeded equal-split run (M = N = 20, T = 600): max 8.6e-16 over 34,066 unit-periods, with 2 capped events and a total shortfall of 0.169.
- Seeded proportional run: max 7.3e-16 over 41,156 unit-periods, with no capped events.

The test requires < 1e-9. The hand-computed cases (2-member, 3-member split by capital, a two-round cascade, a wipeout, a proportional-sharing cap) match to 1e-12. Under C33 the shortfall was not charged to anyone, so the unit held more than the identity.

## Equal-split block (180 runs)

The headline statistics hardly move:

| | C33 | C34 |
|---|---|---|
| mean K | 2.308 | 2.308 |
| pooled capital share | 0.1444 | 0.1442 |
| acceptance rate | 0.1505 | 0.1505 |
| solo+solo / solo+unit / unit+unit acceptance | 0.2153 / 0.1079 / 0.0385 | 0.2152 / 0.1078 / 0.0387 |
| capped members per period | 0.01080 | 0.01068 |
| runs with capped members after burn-in | 101 | 101 |
| shortfall per period | — | 0.00079 (max 0.0069) |

Capped members and the shortfall rise steeply with α: the shortfall is 0.00003 per period at α = 0.1 and 0.0038 at α = 0.5, in capital units with market mean firm size 1. The largest change in any α row is K at α = 0.5 (2.427 → 2.422) and pooled share (0.186 → 0.184). By α:

| alpha | K (mean) C33 → C34 | pooled capital share C33 → C34 | acceptance rate C33 → C34 | acc. solo+solo C33 → C34 | acc. solo+unit C33 → C34 | acc. unit+unit C33 → C34 | capped members / period C33 → C34 | shortfall / period (C34) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | n/a → n/a | 0.000 → 0.000 | 0.0000 → 0.0000 | 0.0000 → 0.0000 | n/a → n/a | n/a → n/a | 0.00000 → 0.00000 | 0.00000 |
| 0.01 | 2.112 → 2.112 | 0.082 → 0.082 | 0.1024 → 0.1024 | 0.1305 → 0.1305 | 0.0678 → 0.0678 | 0.0149 → 0.0149 | 0.00000 → 0.00000 | 0.00000 |
| 0.02 | 2.160 → 2.160 | 0.121 → 0.121 | 0.1481 → 0.1481 | 0.2051 → 0.2051 | 0.0815 → 0.0815 | 0.0253 → 0.0253 | 0.00000 → 0.00000 | 0.00000 |
| 0.05 | 2.233 → 2.233 | 0.155 → 0.155 | 0.1725 → 0.1725 | 0.2537 → 0.2537 | 0.1050 → 0.1050 | 0.0319 → 0.0319 | 0.00003 → 0.00003 | 0.00000 |
| 0.1 | 2.302 → 2.303 | 0.176 → 0.176 | 0.1898 → 0.1896 | 0.2774 → 0.2771 | 0.1213 → 0.1212 | 0.0443 → 0.0442 | 0.00100 → 0.00095 | 0.00003 |
| 0.2 | 2.364 → 2.366 | 0.192 → 0.192 | 0.1979 → 0.1977 | 0.2886 → 0.2885 | 0.1263 → 0.1258 | 0.0470 → 0.0476 | 0.00627 → 0.00623 | 0.00032 |
| 0.3 | 2.392 → 2.391 | 0.195 → 0.195 | 0.1917 → 0.1915 | 0.2789 → 0.2783 | 0.1227 → 0.1227 | 0.0463 → 0.0465 | 0.01607 → 0.01573 | 0.00103 |
| 0.4 | 2.417 → 2.416 | 0.192 → 0.192 | 0.1814 → 0.1819 | 0.2612 → 0.2619 | 0.1172 → 0.1174 | 0.0454 → 0.0456 | 0.02857 → 0.02793 | 0.00198 |
| 0.5 | 2.427 → 2.422 | 0.186 → 0.184 | 0.1710 → 0.1709 | 0.2420 → 0.2418 | 0.1113 → 0.1110 | 0.0436 → 0.0440 | 0.04527 → 0.04520 | 0.00378 |

- C33: 101 of 180 runs with capped members after burn-in; mean capped members per period 0.01080
- C34: 101 of 180 runs with capped members after burn-in; mean capped members per period 0.01068; mean shortfall per period 0.00079 (capital units, market mean firm size = 1)

## Endogenous-α block (60 runs)

Demeaning moves the adopted rate up and makes α = 1 rarer:

| | C33 | C34 |
|---|---|---|
| adopted α, mean | 0.728 | 0.803 |
| adopted α, median of run medians | 0.800 | 0.800 |
| share of units at α = 1 | 0.068 | 0.038 |
| K | 11.72 | 12.09 |
| pooled capital share | 0.905 | 0.922 |
| Hill | 1.673 | 1.728 |

Under realized levels, a member with a lucky window vetoed moves to more pooling. Under demeaning that member consents.

The correlation of α with member volatility weakens but stays positive and clear (Spearman ρ of α_tavg with the mean member IQR, by family; full tables below):
- normal: 0.363 → 0.260
- laplace: 0.320 → 0.194
- t3: 0.240 → 0.139

All three have p < 1e-18.

| log_family | α mean C33 → C34 | α median C33 → C34 | share at α = 1 C33 → C34 | K (mean) C33 → C34 | pooled capital share C33 → C34 | Hill C33 → C34 |
|---|---|---|---|---|---|---|
| laplace | 0.734 → 0.805 | 0.785 → 0.825 | 0.060 → 0.027 | 12.26 → 12.66 | 0.913 → 0.928 | 1.737 → 1.796 |
| normal | 0.755 → 0.818 | 0.793 → 0.840 | 0.077 → 0.027 | 13.80 → 14.35 | 0.909 → 0.926 | 1.719 → 1.796 |
| t3 | 0.695 → 0.786 | 0.750 → 0.800 | 0.068 → 0.059 | 9.09 → 9.25 | 0.894 → 0.910 | 1.561 → 1.593 |

| family | n units C33 → C34 | Spearman ρ(α_tavg, mean member IQR) C33 → C34 | p (C34) |
|---|---|---|---|
| laplace | 3323 → 3353 | 0.320 → 0.194 | 8.97e-30 |
| normal | 3244 → 3210 | 0.363 → 0.260 | 6.39e-51 |
| t3 | 4124 → 4166 | 0.240 → 0.139 | 2.51e-19 |

- C33: block mean α 0.728, median of run medians 0.800, share at α = 1 0.068, K 11.72, pooled capital share 0.905, Hill 1.673
- C34: block mean α 0.803, median of run medians 0.800, share at α = 1 0.038, K 12.09, pooled capital share 0.922, Hill 1.728

## Main-block t₃ runs with capped members in C33

The 8 main-block t₃ runs with capped members after burn-in (from the C33 tidy: 371, 381, 459, 461, 464, 465, 467, 471) are unchanged at the printed precision. Under proportional sharing the cap binds by at most Φ(K)·s_i, so the shortfall is small: between 4e-9 and 6e-6 per period, in capital units. Their pickles differ from C33 in the last bits for 6 of the 8; 371 and 381 are byte-identical.

| id | block | family | cost | α | K C33 → C34 | pooled share C33 → C34 | Hill C33 → C34 | capped / period C33 → C34 |
|---|---|---|---|---|---|---|---|---|
| 371 | main | t3 | linear | 0.02 | 2.330 → 2.330 | 0.3579 → 0.3579 | 0.9793 → 0.9793 | 0.00033 → 0.00033 |
| 381 | main | t3 | linear | 0.1 | 2.817 → 2.817 | 0.5584 → 0.5584 | 0.9996 → 0.9996 | 0.00033 → 0.00033 |
| 459 | main | t3 | exponential | 0.01 | 2.058 → 2.058 | 0.0142 → 0.0142 | 0.9858 → 0.9858 | 0.00067 → 0.00067 |
| 461 | main | t3 | exponential | 0.02 | 2.081 → 2.081 | 0.0416 → 0.0416 | 1.0191 → 1.0191 | 0.00033 → 0.00033 |
| 464 | main | t3 | exponential | 0.02 | 2.108 → 2.108 | 0.0651 → 0.0651 | 0.9878 → 0.9878 | 0.00033 → 0.00033 |
| 465 | main | t3 | exponential | 0.05 | 2.420 → 2.420 | 0.2217 → 0.2217 | 0.9867 → 0.9867 | 0.00033 → 0.00033 |
| 467 | main | t3 | exponential | 0.05 | 2.385 → 2.385 | 0.2153 → 0.2153 | 0.9943 → 0.9943 | 0.00033 → 0.00033 |
| 471 | main | t3 | exponential | 0.1 | 2.725 → 2.725 | 0.3444 → 0.3444 | 1.0153 → 1.0153 | 0.00033 → 0.00033 |

## Identity report (`diagnostics/identity_c34.csv`, `diagnostics/identity_c34b_c34c.csv`)

All 1,800 runs were rerun, not only the ~250 targeted ones. A run is unchanged only if no member is capped at any step, burn-in included. The C33 tidy counts capped members only after burn-in, so the targeted list would have missed 17 main-block runs and 3 search-block runs whose capped members all fall in burn-in. The full rerun cost about 102 CPU-hours.

Comparison: every pickled field except the runtime (`elapsed_seconds`, `ms_per_step`) as `pickle.dumps` bytes, with the summary dict on C33's keys, plus the tidy row as CSV text without the runtime and C34 columns.

- **No capped member at any step, outside the endogenous-α block:** 1,606 of 1,606 byte-identical to C33.
- **Capped members at some step, outside the endogenous-α block:** 134 runs, of which 8 are still byte-identical (367, 371, 377, 381, 438, 1505, 1506, 1528). Two were inspected:
  - In 438 every member was capped (below).
  - In 371 both members lost almost everything in the same period, so the member charged with the shortfall was below the floor anyway.
  The other 126 differ: 106 equal-split, 20 main.
- **Endogenous-α block:** 60 of 60 differ, as expected from C34b.
- **Traced runs:** the 27 traced runs (main, power_law, normal/laplace/t3 × α ∈ {0.05, 0.1, 0.3} × reps 0–2) have no capped member at any step and are byte-identical to C33. Traces, status panel, event study and burstiness are therefore unchanged and were not regenerated.
- **Rename (C34b vs C34c):** 275 full-T runs, byte-identical in every field including the C34 counters. They cover all 180 equal-split and all 60 endogenous-α runs, the 8 main t₃ runs with capped members, and 3 runs from every other block (main 11, rule-replay 3, correlation 3, size 3, …).

## Wipeout (the brief expected none)

Scenario 438 (main, t₃, quadratic, α = 0.3) has one wipeout. At step 102, in burn-in, both members of a two-member unit drew log relative returns of −16.0 and −8.2 in the same period. The unit's gross return of 8.4e-5 is below Φ(2) = 1.06e-4, so its intended capital is negative and both members are capped. C33 capped both members of that unit in exactly the same way, so the run is byte-identical to C33.

`run_pilot_c.py` asserted zero wipeouts and stopped this run on the first pass. It now reports them instead: a warning, `summary['pool_wipeout_total']`, and the tidy column `pool_wipeout_total` (1 in 438, 0 elsewhere). Events are in `diagnostics/capped_events_c34.csv`.

## Other decisions

- **Decision replay.** It keeps −∞ for a replayed p ≤ −1 and does not model the redistribution of a shortfall; MODEL_FLAGS.md documents this. The brief's open question (cap-and-redistribute in the replay as well) was not acted on.
- **`mu_range`.** It is still used when `g` is None, which in the pilot means only the 45 `rule-replay` runs. Setting their location to `g` would change their bits, although under `market_size_fixed` a per-market location cancels from r̃ up to rounding.
- **Stored schema not renamed.** That includes the `hyperparameters` and `summary` keys, so `summarize_pilot_c.py` needs no mapping. The rename table is in MODEL_FLAGS.md.
- **Stale import path.** `tests/test_assortativity.py` had put a stale `/groups/.../IOxEE` tree first on `sys.path`, so four tests ran that tree's `test_phase_b_identity`. Fixed, with a guard test.
- **Data:** `/scratch/bt307958/c34/IOxEE`, persisted to `/groups/m-larch/bt307958/IOxEE_c34` (1,800 results plus the 275 C34b reference results in `pilot_c/results_c34b_ref`). Checksum comparison: 0 files differ. The C33 results stay in `/groups/m-larch/bt307958/IOxEE_c33`.
