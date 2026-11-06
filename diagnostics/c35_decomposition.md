# C35 Task 1 — ordinary-period growth versus ruin protection in the owner-return gap

Source: `pilot_c/status_panel_decomposition.csv` (one row per trace). Code: `analysis/status_panel.decompose`,
`analysis/run_decomposition.py`. Input: the 27 traces of the main-block power-law cells (normal, laplace, t₃ ×
α ∈ {0.05, 0.1, 0.3} × reps 0–2; `/scratch/bt307958/traces_c29`). All firm-periods after burn-in (t ≥ 8000).

**Checks.** In all 27 traces, the floor hit recorded in `floor[i, t]` is exactly the one that sets `jump[t+1, i] ≠ 0`.
The brief's "floor[i, t+1] = 0" means no floor recapitalisation between t and t+1, which is `floor[:, t]` in the
(F, T) array. The per-step floor-hit totals of every trace equal those of the C34 result pickle of the same
scenario, so the C29 traces are the C34 runs. The decomposition cells are the cells of the existing panel: the
decomposition total equals `diff_fe_mean` to 0.0e+00, and A + B + C equals the total to 1.7e-18.

## Decomposition

Inside each (market, size-decile) cell, p is the floor rate and m the mean next-period owner return, by status
(m = member, s = standalone) and sample (nf = no floor in t+1, f = floor in t+1):

total = mean_m − mean_s = **A** (1 − p_s)(m_m,nf − m_s,nf) + **B** [p_m·m_m,f − p_s·m_s,f] + **C** (p_s − p_m)·m_m,nf

- **A** is the ordinary-period term (drag reduction in periods without recapitalisation).
- **B + C** = p_m(m_m,f − m_m,nf) − p_s(m_s,f − m_m,nf) is the part due to members reaching the floor less often
  (ruin protection).
- Cells are aggregated with the panel's weights (member firm-periods per cell).
- share_ordinary = A / total and share_ruin = (B + C) / total.
- The no-floor and floor gaps (columns 5–7) use the panel's fixed-effects construction on each subsample.

The table gives means over the three reps, with the min and max of share_ordinary over the reps. Floor rates are
over all firm-periods; inside the cells where members are, they are much closer, e.g. 0.012 versus 0.020 for
normal at α = 0.3 (`p_floor_*_cells`).

| family | α | total gap (mean) | total gap (median) | no-floor gap (mean) | no-floor gap (median) | floor gap (mean) | A ordinary | B floor periods | C composition | share ordinary [min, max over reps] | share ruin | floor rate m / s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laplace | 0.05 | 0.00103 | 0.00015 | -0.00061 | -0.00077 | 0.00008 | -0.00056 | 0.00139 | 0.00020 | -0.56 [-0.84, -0.35] | 1.56 | 0.027 / 0.189 |
| laplace | 0.1 | 0.00272 | 0.00084 | 0.00022 | -0.00048 | 0.00898 | 0.00027 | 0.00223 | 0.00022 | 0.10 [0.05, 0.13] | 0.90 | 0.025 / 0.196 |
| laplace | 0.3 | 0.00815 | 0.00378 | 0.00292 | 0.00123 | 0.04137 | 0.00308 | 0.00483 | 0.00024 | 0.38 [0.37, 0.39] | 0.62 | 0.017 / 0.206 |
| normal | 0.05 | 0.00054 | -0.00003 | -0.00032 | -0.00067 | -0.00016 | -0.00029 | 0.00072 | 0.00012 | -0.63 [-1.04, -0.19] | 1.63 | 0.020 / 0.152 |
| normal | 0.1 | 0.00131 | 0.00045 | -0.00000 | -0.00050 | 0.00627 | 0.00004 | 0.00113 | 0.00014 | 0.02 [-0.15, 0.17] | 0.98 | 0.018 / 0.162 |
| normal | 0.3 | 0.00427 | 0.00242 | 0.00160 | 0.00047 | 0.02680 | 0.00177 | 0.00231 | 0.00018 | 0.41 [0.36, 0.45] | 0.59 | 0.012 / 0.173 |
| t3 | 0.05 | 0.00266 | 0.00028 | -0.00065 | -0.00070 | 0.03258 | -0.00059 | 0.00305 | 0.00020 | -0.23 [-0.28, -0.16] | 1.23 | 0.032 / 0.201 |
| t3 | 0.1 | 0.00452 | 0.00067 | -0.00042 | -0.00081 | 0.04517 | -0.00035 | 0.00466 | 0.00021 | -0.08 [-0.10, -0.05] | 1.08 | 0.030 / 0.207 |
| t3 | 0.3 | 0.01136 | 0.00358 | 0.00190 | 0.00061 | 0.06972 | 0.00205 | 0.00906 | 0.00026 | 0.18 [0.17, 0.20] | 0.82 | 0.021 / 0.217 |

## What fraction of the gain is ordinary-period drag reduction

**At α = 0.3**, ordinary-period growth accounts for:

| family | share of the total gap from ordinary periods | the rest (lower floor rate of members) |
|---|---|---|
| normal | 41% (reps 36–45%) | 59% |
| laplace | 38% | 62% |
| t₃ | 18% | 82% |

So the expected "about half under normal, a third under t₃" holds in direction but overstates the ordinary
component, under t₃ by almost half. The heavier the tail, the more of the gain is ruin protection.

**At α ≤ 0.1**, the ordinary-period term is about zero or negative, so the whole gain, and more, comes from
members reaching the floor less often:

- normal and laplace at α = 0.1: share_ordinary is 0.02 and 0.10;
- at α = 0.05 and under t₃ at α = 0.1: A < 0, and the shares range from −0.08 to −0.63.

In ordinary periods members then grow slightly *less* than standalones in the same cell. A small sharing rate
does little to the body of the distribution, while the coordination cost Φ(K) and the members' proportional
share of the pool are paid every period.

**How to read A.** The split is an exact accounting identity, but the no-floor sample is selected on the outcome.
Inside a cell, the floor removes more of the standalones' lower tail (p_s > p_m), which raises their no-floor
mean relative to members'. A should therefore be read as a conservative measure of the ordinary-period
component, not as a causal drag-reduction effect.

**Floor periods themselves** are also less severe for members (floor gap +0.027 to +0.070 at α = 0.3).
Term B carries both the frequency effect and this severity effect.

**Medians.** At α = 0.3 the median gap is 32–57% of the mean gap (t₃ lowest, normal highest), and much smaller at
low α. This is
consistent with a gain concentrated in rare, large-loss periods.
