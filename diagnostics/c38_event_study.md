# C38 Task 1: why the owner-path DiD is negative under the moment rule

Branch `referee-fixes`, from C37 (`b4b9ea2`). No model run. This is the event study on the 27 existing pilot D
traces (`pilot_d/traces/`, on Festus): main block, power law, three families × α ∈ {0.05, 0.1, 0.3} × reps 0–2,
moment rule.

**How it was run.**
- Script: `analysis/event_study.py` gets two flags, `match` and `hitfree_k`. Its defaults reproduce C29–C37.
- Driver: `analysis/run_traces.py --matching` / `--merge-matching`.
- Outputs:
  - `pilot_d/event_study_matching.csv`: means over the 3 reps, by family × α × matching;
  - `pilot_d/event_study_matching_by_rep.csv`: per trace.
- **Check.** The current matching reproduces `pilot_d/event_study_summary.csv` (C37) on all 27 traces: the
  matched count and the ITT and owner DiD medians. The merge stops otherwise.

**Units and terms.**
- All figures are in pp of log market share per period, i.e. the window change divided by l = 50.
- **Owner path:** the log share net of the cumulative recapitalisation at the floor.
- **Pre gap:** mean over matched events of (joiner − control) owner-path change in the 50 periods before entry.
- **Post gap:** the same for the 50 periods after entry.
- **Means are additive:** DiD owner mean = post gap − pre gap. The medians in the C37 note do not decompose.

## 1. Decomposition under the current matching (raw log share at entry)

The table shows the mean owner-path change over matched events for the joiner (J) and the control (C).

| family | α | J before | C before | J after | C after | pre gap | post gap | DiD owner mean | recap J − C, before |
|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | −2.36 | −2.41 | −1.01 | −1.02 | +0.05 | +0.01 | −0.04 | −0.55 |
| normal | 0.1 | −2.33 | −2.42 | −0.95 | −0.97 | +0.09 | +0.02 | −0.07 | −0.57 |
| normal | 0.3 | −2.09 | −2.30 | −0.71 | −0.80 | +0.22 | +0.10 | −0.12 | −0.61 |
| laplace | 0.05 | −3.59 | −3.79 | −1.86 | −1.90 | +0.21 | +0.04 | −0.16 | −0.82 |
| laplace | 0.1 | −3.54 | −3.77 | −1.74 | −1.79 | +0.23 | +0.04 | −0.19 | −0.84 |
| laplace | 0.3 | −3.21 | −3.62 | −1.36 | −1.54 | +0.41 | +0.17 | −0.23 | −0.92 |
| t₃ | 0.05 | −4.34 | −4.62 | −2.40 | −2.46 | +0.28 | +0.06 | −0.22 | −0.88 |
| t₃ | 0.1 | −4.26 | −4.61 | −2.26 | −2.35 | +0.35 | +0.08 | −0.27 | −0.95 |
| t₃ | 0.3 | −3.93 | −4.52 | −1.86 | −2.08 | +0.59 | +0.22 | −0.37 | −1.11 |

"Recap J − C, before" is the joiner's recapitalisation in the pre-window minus the control's, per period. It equals
the raw-path pre gap minus the owner-path pre gap.

**The pre-period carries the sign.**
- **Both gaps are positive in every cell.**
  - Post gap: +0.01 to +0.22.
  - Pre gap: +0.05 to +0.59, about two to six times as large.
  - The DiD is post minus pre, so it is negative.
- **Rep by rep.** The pre gap is positive in 27/27 traces and the post gap in 25/27. The two exceptions are normal
  α = 0.05 and α = 0.1, each at −0.006 pp or less.

**The pre gap is positive because of the controls, not the joiners.**
- Matching on raw share at entry pairs the joiner with a control that was **recapitalised more** before entry: by
  0.55 to 1.11 pp per period.
- **Raw path.** The control's raw share **rose** before entry (+0.04 to +0.17 pp per period) while the joiner's
  fell (−0.25 to −0.54).
- **What the matching picks.** A standalone at the same raw share as a joiner tends to be one just topped up by
  the floor. Net of those top-ups, its owner path fell further than the joiner's.
- **The C37 note had the sign the wrong way.** It said that the joiners' pre-entry hits put a drop on the
  joiner's owner path. A drop on the joiner's side would make the DiD *larger*. What makes it negative is the
  controls' larger recapitalisation.

## 2. The four matchings

The table shows means over the 3 reps.
- **events:** event count. Under B it is the joiners that survive the hit-free condition.
- **matched:** events with a control within the calliper.

| family | α | matching | events | matched | DiD ITT median | DiD owner median | DiD owner mean | pre gap (owner) | post gap (owner) |
|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | current (raw share) | 383693 | 199608 | +0.40 | −0.01 | −0.04 | +0.05 | +0.01 |
| normal | 0.05 | A: owner-path share | 383693 | 103324 | +0.44 | +0.39 | +0.43 | −0.42 | +0.01 |
| normal | 0.05 | B: hit-free k = 5 | 247961 | 94822 | +0.36 | +0.15 | +0.14 | −0.13 | +0.01 |
| normal | 0.05 | B: hit-free k = 10 | 175546 | 56151 | +0.28 | +0.16 | +0.16 | −0.15 | +0.01 |
| normal | 0.05 | B: hit-free k = 20 | 106626 | 25871 | +0.19 | +0.15 | +0.16 | −0.14 | +0.02 |
| normal | 0.05 | C: same hit count | 383693 | 19838 | +0.29 | +0.33 | +0.36 | −0.33 | +0.03 |
| normal | 0.1 | current (raw share) | 350224 | 177284 | +0.39 | −0.03 | −0.07 | +0.09 | +0.02 |
| normal | 0.1 | A: owner-path share | 350224 | 90550 | +0.44 | +0.39 | +0.44 | −0.41 | +0.03 |
| normal | 0.1 | B: hit-free k = 5 | 220117 | 82412 | +0.34 | +0.16 | +0.14 | −0.11 | +0.02 |
| normal | 0.1 | B: hit-free k = 10 | 151730 | 48120 | +0.26 | +0.17 | +0.17 | −0.14 | +0.03 |
| normal | 0.1 | B: hit-free k = 20 | 87328 | 21678 | +0.17 | +0.17 | +0.16 | −0.12 | +0.04 |
| normal | 0.1 | C: same hit count | 350224 | 16961 | +0.28 | +0.32 | +0.36 | −0.33 | +0.03 |
| normal | 0.3 | current (raw share) | 246713 | 113320 | +0.33 | −0.06 | −0.12 | +0.22 | +0.10 |
| normal | 0.3 | A: owner-path share | 246713 | 58159 | +0.38 | +0.38 | +0.41 | −0.32 | +0.10 |
| normal | 0.3 | B: hit-free k = 5 | 150488 | 53113 | +0.28 | +0.15 | +0.15 | −0.03 | +0.12 |
| normal | 0.3 | B: hit-free k = 10 | 101061 | 31671 | +0.20 | +0.19 | +0.18 | −0.07 | +0.11 |
| normal | 0.3 | B: hit-free k = 20 | 55686 | 14684 | +0.12 | +0.17 | +0.17 | −0.07 | +0.10 |
| normal | 0.3 | C: same hit count | 246713 | 10984 | +0.22 | +0.31 | +0.33 | −0.23 | +0.10 |
| laplace | 0.05 | current (raw share) | 501805 | 238524 | +0.49 | −0.12 | −0.16 | +0.21 | +0.04 |
| laplace | 0.05 | A: owner-path share | 501805 | 104984 | +0.53 | +0.49 | +0.54 | −0.51 | +0.03 |
| laplace | 0.05 | B: hit-free k = 5 | 303763 | 95167 | +0.45 | +0.14 | +0.12 | −0.08 | +0.04 |
| laplace | 0.05 | B: hit-free k = 10 | 204715 | 50951 | +0.37 | +0.15 | +0.15 | −0.13 | +0.02 |
| laplace | 0.05 | B: hit-free k = 20 | 115805 | 20705 | +0.28 | +0.16 | +0.16 | −0.15 | +0.01 |
| laplace | 0.05 | C: same hit count | 501805 | 19601 | +0.34 | +0.44 | +0.49 | −0.43 | +0.06 |
| laplace | 0.1 | current (raw share) | 466988 | 215777 | +0.49 | −0.13 | −0.19 | +0.23 | +0.04 |
| laplace | 0.1 | A: owner-path share | 466988 | 93916 | +0.52 | +0.48 | +0.54 | −0.50 | +0.04 |
| laplace | 0.1 | B: hit-free k = 5 | 275998 | 84625 | +0.43 | +0.15 | +0.14 | −0.09 | +0.06 |
| laplace | 0.1 | B: hit-free k = 10 | 181137 | 44623 | +0.35 | +0.18 | +0.19 | −0.14 | +0.05 |
| laplace | 0.1 | B: hit-free k = 20 | 97650 | 17653 | +0.24 | +0.19 | +0.21 | −0.16 | +0.05 |
| laplace | 0.1 | C: same hit count | 466988 | 17240 | +0.33 | +0.41 | +0.47 | −0.44 | +0.04 |
| laplace | 0.3 | current (raw share) | 348456 | 150288 | +0.43 | −0.16 | −0.23 | +0.41 | +0.17 |
| laplace | 0.3 | A: owner-path share | 348456 | 65435 | +0.49 | +0.54 | +0.58 | −0.40 | +0.18 |
| laplace | 0.3 | B: hit-free k = 5 | 198435 | 58661 | +0.38 | +0.19 | +0.18 | +0.02 | +0.20 |
| laplace | 0.3 | B: hit-free k = 10 | 126162 | 31469 | +0.29 | +0.25 | +0.26 | −0.06 | +0.20 |
| laplace | 0.3 | B: hit-free k = 20 | 64533 | 12811 | +0.20 | +0.25 | +0.27 | −0.08 | +0.20 |
| laplace | 0.3 | C: same hit count | 348456 | 11845 | +0.27 | +0.41 | +0.45 | −0.29 | +0.17 |
| t₃ | 0.05 | current (raw share) | 565695 | 275202 | +0.47 | −0.13 | −0.22 | +0.28 | +0.06 |
| t₃ | 0.05 | A: owner-path share | 565695 | 111857 | +0.46 | +0.47 | +0.53 | −0.49 | +0.04 |
| t₃ | 0.05 | B: hit-free k = 5 | 335960 | 102475 | +0.44 | +0.14 | +0.11 | −0.04 | +0.07 |
| t₃ | 0.05 | B: hit-free k = 10 | 223790 | 52345 | +0.37 | +0.19 | +0.18 | −0.11 | +0.07 |
| t₃ | 0.05 | B: hit-free k = 20 | 125299 | 20104 | +0.29 | +0.23 | +0.25 | −0.17 | +0.08 |
| t₃ | 0.05 | C: same hit count | 565695 | 22418 | +0.32 | +0.42 | +0.46 | −0.38 | +0.09 |
| t₃ | 0.1 | current (raw share) | 531761 | 254662 | +0.47 | −0.15 | −0.27 | +0.35 | +0.08 |
| t₃ | 0.1 | A: owner-path share | 531761 | 101684 | +0.48 | +0.50 | +0.56 | −0.49 | +0.07 |
| t₃ | 0.1 | B: hit-free k = 5 | 308471 | 92502 | +0.45 | +0.16 | +0.12 | −0.01 | +0.11 |
| t₃ | 0.1 | B: hit-free k = 10 | 200365 | 46277 | +0.37 | +0.22 | +0.21 | −0.10 | +0.11 |
| t₃ | 0.1 | B: hit-free k = 20 | 107550 | 17282 | +0.27 | +0.22 | +0.23 | −0.15 | +0.07 |
| t₃ | 0.1 | C: same hit count | 531761 | 20022 | +0.31 | +0.43 | +0.42 | −0.35 | +0.07 |
| t₃ | 0.3 | current (raw share) | 413804 | 192534 | +0.43 | −0.23 | −0.37 | +0.59 | +0.22 |
| t₃ | 0.3 | A: owner-path share | 413804 | 76257 | +0.47 | +0.56 | +0.62 | −0.40 | +0.21 |
| t₃ | 0.3 | B: hit-free k = 5 | 231339 | 68384 | +0.39 | +0.16 | +0.12 | +0.14 | +0.26 |
| t₃ | 0.3 | B: hit-free k = 10 | 145366 | 34229 | +0.31 | +0.23 | +0.24 | +0.01 | +0.26 |
| t₃ | 0.3 | B: hit-free k = 20 | 73323 | 12755 | +0.22 | +0.27 | +0.29 | −0.07 | +0.22 |
| t₃ | 0.3 | C: same hit count | 413804 | 14745 | +0.26 | +0.39 | +0.39 | −0.20 | +0.19 |

**Events that survive the hit-free window (B).**

| k | share of the C37 events that survive |
|---|---|
| 5 | 56–65% |
| 10 | 35–46% |
| 20 | 18–28% |

Matched events per trace (mean over reps) stay above 12,700 at every k. Matching C matches only 3.6–5.2% of events
(10,984–22,418 per trace), because the hit count must agree exactly.

**Signs, trace by trace (27 traces).**

| matching | owner DiD (median and mean) positive | pre gap positive | post gap positive |
|---|---|---|---|
| current | 0 | 27 | 25 |
| A | 27 | 0 | 25 |
| B, k = 5 | 27 | 6 | 26 |
| B, k = 10 | 27 | 3 | 25 |
| B, k = 20 | 27 | 0 | 26 |
| C | 27 | 0 | 26 |

The post gaps that are not positive are all at α ≤ 0.1 (normal and laplace) and at most 0.01 pp in absolute value.

## 3. What the negative sign is

**Every alternative matching removes the sign.**
- Owner DiD medians across cells:

  | matching | owner DiD median |
  |---|---|
  | current | −0.01 to −0.23 |
  | A | +0.38 to +0.56 |
  | B | +0.14 to +0.27 |
  | C | +0.31 to +0.44 |

- **Every matching moves only the pre gap.** In each cell the post-entry gap is nearly the same under all six
  matchings:

  | cell | post gap, all matchings |
  |---|---|
  | normal α = 0.05 | +0.01 to +0.03 |
  | normal α = 0.3 | +0.10 to +0.12 |
  | laplace α = 0.3 | +0.17 to +0.20 |
  | t₃ α = 0.3 | +0.19 to +0.26 |

  It rises with α and with tail weight.
- **The pre gap swings with the matching.** It runs from +0.05 to +0.59 under raw-share matching to −0.20 to −0.51
  under matchings A and C.

**So the negative sign is a pre-trend from the comparison group.** It is not a post-entry effect.
- **Under raw-share matching** the controls are firms whose raw share was being propped up by recapitalisation.
  That is a composition effect in the control group, and it shows as a pre-trend on the owner path.
- **Matching C is the brief's test of the C37 explanation.** It shows recapitalisation history is the cause, since
  equalising the hit count flips the sign. But the culprit is the controls' history, not the joiners'.

**No matching gives a clean DiD.**
- **A and C overshoot the other way.** Once the controls share the joiners' owner-path level or hit count, the
  joiners' owner path fell faster than the controls' before entry (pre gap −0.20 to −0.51). This is a dip before
  entry, so DiD owner overstates the effect (means +0.33 to +0.62).
- **B, the hit-free window, comes closest to parallel pre-trends.**
  - Pre gap: −0.17 to +0.14.
  - Its DiD (+0.11 to +0.29 by mean) is closest to the post gap.

**The quantity that does not depend on the matching is the post-entry gap.** On the owner path, joiners grow
faster than matched standalones after entry, by 0.01 to 0.26 pp per period, increasing in α.

**For the paper.**
- Proposition 3 can keep resting on the status panel and the decomposition (C37).
- The event study can be reported in two forms, both positive in every cell:
  - the hit-free matching (B, k = 10 or 20);
  - the post-entry gap.
- The C37 note's sentence on the cause (joiners' pre-entry floor hits) should be corrected along the lines of
  Section 1.
