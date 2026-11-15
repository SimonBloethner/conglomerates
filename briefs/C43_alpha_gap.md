# C43 — Main block at α ∈ {0.6, 0.7, 0.8}

**Why.** The main block's sharing-rate grid ends at 0.5; the only cells beyond are the fixed α = 0.9 block (C38)
and the endogenous block at its adopted rate. Figures 1 and 4 (scope and Hill exponent against α) and the sentence
"about 0.45 per unit of α up to 0.5 and faster beyond" therefore rest on one point past 0.5. This fills the gap.
Run time: main-block runs take ~230 s at α = 0.5 and ~470 s at α = 0.9, so expect 5–7 minutes each; 45 runs in
all, parallel.

**Design.** Append one block to `pilot_design_d.py`, following `generate_fixed_alpha` exactly:

- block name `alpha-gap`; scenario ids 2100–2144 (the 2,100 existing scenarios are unchanged)
- the main power-law scenario of each family (normal, laplace, t3) and rep 0–4, copied as the α = 0.9 block copies
  the α = 0.5 scenario (same cell id, same seed, same parameters), with `alpha` set to 0.6, 0.7 and 0.8
- moment rule only; `twin_id` and `source_id` None; no loggain twin, no cv or accept twin
- update the module docstring's block list and the counts in `tests/test_pilot_d_design.py` (2,145 scenarios,
  moment 2,055 incl. α = 0.9 and alpha-gap)

**Run.** The 45 new scenarios on the same code and settings as the main block (T = 11,000, retained window
8,000–11,000, the same output fields and counters). Nothing else is rerun.

**Identity check.** Regenerate `pilot_d/scenarios.json`; every one of the 2,100 existing scenarios must be
byte-identical to the stored file apart from nothing (no field changes), and the 15 α = 0.9 scenarios must still
match their stored results when re-summarised. Report the count as in C42 ("2,100/2,100").

**Outputs.**

1. `pilot_d/tidy.csv` extended by the 45 rows (same columns; `block = alpha-gap`).
2. `pilot_d/fixed_alpha_09.csv` left as is; a new `pilot_d/alpha_gap.csv` with, per family × α × rep: K_mean,
   cong_capital_share_median, hill_exponent_median, hhi_within_median, floor_exits_per_period,
   voluntary_exits_per_period, acceptance_rate, floor_hit_rate_member_status, floor_hit_rate_standalone_status,
   top10pct_aggregate_median, K_eff_over_K_median, elapsed_seconds — the columns Tables 3, A18 and Figures 1 and 4
   draw on.
3. A short results note (`pilot_d/alpha_gap_report.md`): the medians over reps for each family × α beside the
   α = 0.5 and α = 0.9 cells, nothing else. No reading of the result; the paper text is adjusted afterwards from
   the numbers.

**What this must not do.** No change to the model code, the floor, the cost, the seeds of the existing cells or the
summary schema. If any existing scenario changes under the regenerated design, stop and report rather than
re-run.

**After the results are on the repo (my side).** `paper_figures.py`: Figures 1 and 4 plot the main block and
`alpha-gap` as one curve, the shaded "no runs" band goes. `paper_tables.py`: Table 3 (appendix) gains the three
rows per family; Table A18 likewise. Section 4.4 and 4.6: the "faster beyond 0.5" sentence is rewritten from the
numbers, whichever way they come out; Section 4.4's "going from α = 0.5 to 0.9 doubles scope" likewise.
