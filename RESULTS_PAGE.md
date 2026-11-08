# Results: Conglomerate Mergers under Multiplicative Growth — v2 (pilot D, moment rule as base case)

Model: `collaborative_growth.py` at `ecd6273` (C36a: C34 model plus the moment decision rule); design `pilot_design_d.py`; every number below from `pilot_d/` at `4237629` (C38c) unless marked (loggain: `pilot_c/`, same seeds). Notes: `diagnostics/c33_*`, `c34_note.md`, `c35_note.md`, `c36_note.md`, `c37_note.md`, `c38_note.md`, `c38_event_study.md`. Figures: `analysis/paper_figures.py` → `figures/paper/`. Supersedes RESULTS_PAGE v1 (C29/02ce733): every v1 number that changed is listed in the last section.

## Reader's guide
Notation follows ergodicity economics: time-average growth g (common to all firms, 0.055, cancels from shares), ensemble rate μ = g + log E[e^{σε}], the gap the volatility premium. Sizes are shares of market capital. Unless stated otherwise: main block, power-law cost, proportional sharing, **moment decision rule**, α = 0.3 the reference cell, K = mean members per active unit over post-burn-in snapshots (K_mean), medians over five replications. Floor-hit rates are **per firm-period of that status** (member per member-period, standalone per standalone-period); the all-firm definition in `tidy.csv` is not used in the text. The loggain rule is the bounded-rationality variant and is quoted beside the base case; the replay rule is out of the paper.

## The model in one page
M = N = 50, T = 11,000, first 8,000 discarded (16,000-period rerun: no drift). Order within a period: shocks → relative returns r̃ = x − log G → [endogenous block: α renegotiation] → proposals and consent → pooling (limited-liability shortfall borne by the unit's other members, C34) → floor (fixed point of c × mean; hit = recapitalization of a going concern, record kept, affiliation ends) → renormalization → exit review (tenure > l).

**Shocks.** x = g + σₘε, ε normal / Laplace / t₃ at equal IQR qₘ ∈ [0.10, 0.30]; σₘ = qₘ/(2F⁻¹(0.75)).

**Pooling.** δᵢ = (1 − α)e^{r̃ᵢ} + αΣωⱼe^{r̃ⱼ} − Φ(K); Φ power law (main), linear, quadratic, exponential at Φ(40) = 0.0017.

**Decision (moment rule, C36).** On the demeaned window of l = 50 periods, member i's estimated gain from set Ĉ at rate α is
Δ̃ᵢ = ½[(1 − α)V̂(εᵢ) + αΣⱼω̄ⱼV̂(εⱼ) − V̂(yᵢ)] − Φ(K̂), yᵢ = (1 − α)εᵢ + αΣⱼωⱼεⱼ — the replayed log growth expanded to second order; equals ½α(2 − α)σ²(1 − 1/K) for K equal i.i.d. members. Consent iff Δ̃ᵢ(Ĉ) > max{0, Δ̃ᵢ(Cᵢ)} for every i ∈ Ĉ (unanimity, veto for all, fixed terms, no side payments); exit iff Δ̃ᵢ(Cᵢ) < 0 after lock-in l. **Loggain** (variant): the same window and tests with the replayed sample mean of log(1 + p̂ᵢ) − εᵢ. Search: each firm proposes with p = 0.05 to a uniform firm in a uniform unoccupied market; unit–unit mergers as wholes; one firm per market.

**Endogenous α** (block): grid {0, 0.05, …, 1}, unanimity first (every member gains over the window), maximin second; start 0.1; merged unit takes the larger unit's rate.

**Floor.** c* = 0.12717 (N = 50; 0.069 at N = 25, 0.158 at N = 100), calibrated so the unaffiliated Laplace economy has Hill 1.06 (Axtell).

**Design.** 1,995 runs (+15 at α = 0.9, C38): main 540; equal split 180; cost level 360; cost level full 180; cost level high 90; lookback 100; correlation 45; endogenous α 60; floor 90; search 120; size 80; homogeneous (q = 0.20) 120 incl. 60 loggain; ν ∈ {2.5, 4} 30. Every cell except ν also under loggain in pilot C (2,100 runs). Common random numbers across α.

## Baseline (α = 0)
Hill 1.07 (normal), 1.06 (Laplace), 1.00 (t₃; 0.96 at ν = 2.5) — calibration holds for the light-tailed families. Floor-level block: c = 0.10 → 1.05–1.29 across α, c = 0.35 → 1.55–1.80. Within-market HHI 0.17–0.20. Standalone floor-hit rate at α = 0: 8.7 / 11.6 / 12.9% of firm-periods. No "period = year".

## Proposition 1 — Conglomerates form at once, hold most capital, and stay far below K*; survival binds first, then verifiability; the transfer conflict is immaterial; cost only at the margin
**Scope (Table 3).** K at α = 0.05/0.1/0.3/0.5: normal 3.2/3.7/5.5/7.7; Laplace 3.2/3.7/5.4/7.8; t₃ 3.1/3.5/4.8/6.8 (loggain 2.7/3.0/4.3/6.3; 2.7/3.0/4.0/5.9; 2.7/2.9/3.7/5.1). K/K* at α = 0.3: 0.46 / 0.34 / 0.10 (K* = 12 / 16 / 47; t₃ at the cap M = 50 for α ≥ 0.5). Pooled capital 0.76–0.96. The moment rule raises K in 328 of 329 paired cells (exception: search p = 1, normal, α = 0.5, −0.5). Voluntary exits ≈ 0 at α ≥ 0.1 (loggain 0.2–3 per period: the exits were estimator noise). Acceptance 0.27–0.47; pairs 100%, fourth member 59–69%, fifth 48–54%, eighth 20–31%, twelfth 4–13% at α = 0.3.

**Turnover.** Member floor-hit rate 1.4 / 2.2 / 2.8% per member-period at α = 0.3; standalone 19–23% per standalone-period (rising with α as the firms left outside are the riskier ones). Mergers per period = floor exits per period to within 0.2 in every cell (25 / 36 / 42 at α = 0.3). Heavier tails: more willing (acceptance 44% vs 29%), less able to stay pooled.

**Cost.** Φ(5) = 1.4·10⁻⁴ vs gain 0.005–0.012 per period (1–3%); K-th member pays for itself to K ≈ 12/16/17. Level: −0.3 / −0.6 / −1.0 members per doubling at α = 0.1/0.3/0.5 (Laplace), ×0.5–×20; at ×20 K = 2.8 vs K* = 4 (K/K* 0.6–0.7). Shape: < 1 member at equal Φ(40).

**Consent (C35–C38, Figure 6).** The binding member is an incumbent. Half of proposals at K ≥ 4 (58–73% at K = 4–8) bring a floor-sized standalone with unit weight w̄ = 0.06–0.07 at K = 5 (equal member: 0.2) and 0.02–0.03 at K = 8. Incumbent's gain from it: 36–37% of the equal-weight formula at K = 5, 23–26% at K = 8; net margin over marginal cost zero at K ≈ 7 (normal), 8–9 (Laplace, t₃); as a multiple of marginal cost at K = 5: 3.1× / 5.9× / 6.5× (equal-weight: 8.4× / 16× / 18×). Binding mean for floor entrants = that margin − 2–5·10⁻⁵ (minimum over K noisy estimates), crossing zero between K = 5 and 7. Large entrants (w ≥ 0.2; 8–21% of proposals at K = 4–10) have strongly negative binding gains from K = 6: presumably unit–unit proposals (transfer term); not separately verified.

**Window (Table 3b).** Laplace α = 0.3: K 4.8 / 5.4 / 5.8 / 6.1 / 6.6 at l = 20/50/100/200/500 (loggain 3.3 / 4.0 / 4.4 / 4.8 / 5.4); t₃ 4.3 → 5.4 (3.0 → 4.9). Floor exits independent of l; voluntary exits nil at every l under moment.

**Decomposition of K* − K_loggain (Table 3c, C37).** (a) K* − K_hom(moment) = survival + feasibility + recruit size + residual noise: 75–85% (normal), 84–89% (Laplace), 96–97% (t₃). (b) K_hom − K_het = transfer conflict: 0.05–0.41 members, 0–4%. (c) K_moment − K_loggain = verifiability: 0.6–1.8 members, 3–20% (largest share normal, smallest t₃). Caveat: homogeneous economy has mean q² ≈ 8% below heterogeneous.

**Verifiability is bias as well as noise (C38 Task 3).** On the same windows Δ̃ exceeds Δ̂ by a median 20–38%; the omitted higher-order terms are 1–2% (normal/Laplace within ±5–8% in 90% of windows; t₃ −19% to +35%); the rest is the replay's first-order term, negative in most finite demeaned windows (weights high after above-mean returns, which demeaning offsets). Sign disagreement 0.7–2.3% of windows.

## Proposition 2 — The frictionless limit is itself below K*, and the gap widens with the tails
p = 0.05 → 0.2 → 1 at α = 0.3: K 5.5 → 6.6 → 7.4 (normal), 5.4 → 6.8 → 8.0 (Laplace), 4.8 → 6.4 → 7.9 (t₃); pooled 98–99%; mergers and floor exits triple together (25–42 → 78–141); acceptance 29–44% → 8–16%. K/K* at p = 1: 62 / 50 / 17% (α = 0.3), 77 / 56 / 22% (α = 0.5).

## Proposition 3 — Pooling raises owners' time-average growth and cuts the ruin hazard, most under heavy tails; the median hides most of it
Status panel (27 traces, moment): member − standalone owner return, pp per period, and medians at α = 0.3:

| | α = 0.05 | 0.1 | 0.3 | median at 0.3 | by K at 0.3 (2 / 3–4 / 5+) |
|---|---|---|---|---|---|
| normal | 0.07 | 0.16 | 0.51 | 0.32 | 0.28 / 0.42 / 0.54 |
| Laplace | 0.10 | 0.26 | 0.85 | 0.41 | 0.52 / 0.73 / 0.90 |
| t₃ | 0.27 | 0.48 | 1.19 | 0.40 | 0.79 / 1.07 / 1.29 |

Positive in every cell and replication (loggain 0.43 / 0.82 / 1.14 at 0.3); ordered t₃ > Laplace > normal, 2.3:1 at α = 0.3; flat across deciles 2–9, decile 10 at 40–60%. Medians at α = 0.05 within 0.02 pp of zero. **Decomposition (C35 definitions):** ordinary-period share at α = 0.3: 50% / 38% / 22% (ordinary gain 0.26 / 0.32 / 0.26 pp); at α ≤ 0.1 ≈ 0 or negative. **Hazard:** member 1.1 / 1.6 / 2.0% vs standalone 19.5 / 22.3 / 23.2% per status-period at α = 0.3 → ratio 0.055 / 0.072 / 0.086 (18× / 14× / 12×); at α = 0.05: 7–11×. **Burstiness:** index of dispersion 2.4–3.0 (normal), 2.8–3.5 (Laplace), 3.8–4.6 (t₃) vs binomial 0.88–0.94, falling in α; worst market-period 43–49 of 50 at the floor.

**Event study (appendix, C38).** ITT +0.33 to +0.49 pp. Owner-path DiD under raw-share matching −0.01 to −0.23 pp: a pre-trend from the controls (raw-share matching picks standalones just topped up at the floor, recapitalised 0.6–1.1 pp/period more than joiners). Post-entry gap +0.01 to +0.26 pp, stable across matchings; owner-path matching +0.38 to +0.56; hit-free window (k = 10) +0.16 to +0.25; same hit count +0.31 to +0.44 — all 27 traces positive. Report post-entry gap and hit-free matching; the panel is the primary instrument. The C37 note's explanation (joiners' pre-entry hits) was wrong in sign.

## Endogenous sharing rate
Adopted α (power-law cost): mean 0.93 / 0.92 / 0.91, median 0.90 in every family, never 0, 1 in 22 / 14 / 12% of unit-periods (loggain: 0.80–0.85, 1 in 3–5%). K 16.9 / 18.0 / 14.3; pooled 97 / 97 / 95%; Hill 2.05 / 2.07 / 1.83; voluntary exits 0. Exponential cost: K 16–23 at pooled 81–92%. **Fixed α = 0.9 (C38)**: K 16.0 / 17.2 / 13.9, pooled 0.98 / 0.97 / 0.95, Hill 1.98 / 1.94 / 1.78, floor exits 2.4 / 4.9 / 10.5 per period — within 2–6% of the endogenous block: the outcomes are those of the adopted level; renegotiation adds little. Volatility no longer drives the choice: Spearman ρ(α_tavg, mean member IQR) 0.10 / 0.05 / 0.04 (loggain 0.26 / 0.19 / 0.14), ρ with dispersion −0.14 / −0.05 / 0.00; n = 7,630 units (2,468 / 2,319 / 2,843). Symmetric optimum is α = 1 (the v1 anchor α* ≈ K/(K+1) is dropped); an interior rate is unanimity among unequal members. From α = 0.5 to 0.9: K doubles, Hill +0.56–0.64, floor exits ÷3–6.

## Mechanisms
**Sharing rule decides existence.** Equal split (normal, four cost forms): K 2.1–3.2 at every α, pooled ≤ 34%, pair acceptance 3–48% (proportional 95–100%), solo → unit 6–17% (28–41%), voluntary exits 2–5 per period (the exit test now detects cost-driven losses), pool ruin 0.01–0.05 per period at α ≥ 0.3.
**Decision rule.** Moment vs loggain as above; replay out of the paper.
**Memory.** Table 3b above.

## Distribution
Hill 1.06–1.08 (normal, Laplace) and 1.00 (t₃) to α = 0.02 (pooled 51–56%); 1.09–1.10 / 1.01 at α = 0.05 (pooled 76–78%); 1.21 / 1.20 / 1.12 at α = 0.3; 1.34 / 1.34 / 1.22 at α = 0.5; 1.8–2.1 endogenous. Driver is pooled capital, not K (search block: K +35–65%, Hill unchanged, pooled already > 90%). Within-market HHI 0.175–0.20 (α = 0) → 0.14–0.17 (0.3) → 0.11–0.14 (0.5). Top-10% unit share: 0.65 → 0.65 (normal; 0.67 at 0.3), 0.66 → 0.74 (Laplace), 0.68 → 0.79 (t₃). Assortativity 0.60–0.61 (α = 0.01) → 0.35–0.36 (0.5); K_eff/K 0.83–0.88 → 0.64–0.68 (0.68–0.70 at 0.3).

## Robustness (Table 6)
Cost form < 1 member; cost level −0.6 members per doubling (α = 0.3); window 4.8 → 6.6 over l = 20–500; search 50% of K* at p = 1; correlation 0.3: K 5.3 vs 5.4; floor c = 0.10: K +3%, Hill 1.13; c = 0.35: K −19% (−10% to −25% across α), Hill 1.66; size (M, N) ∈ {25, 100}²: K 4.7–6.2, K/K* 0.29–0.38, Hill 1.16–1.24, floor exits 6.5–170 per period; ν = 2.5 / 3 / 4: K 4.1 / 4.8 / 5.3, Hill 1.02 / 1.12 / 1.17, acceptance 0.50 / 0.44 / 0.37, floor exits 54 / 42 / 33 (attrition, not consent); σ̂²/σ² over 50-period windows median 0.59 / 0.77 / 0.89, share < ½ 37% / 13% / 3%. Pool wipeouts: 3 events in 1,995 runs (scenarios 494, 1817, 1967).

## Caveats to state
Fixed market capital (share numeraire). Firms judge on second moments of a demeaned window; the moment statistic is a second-order approximation (omitted terms 1–2% median, heavy-tailed under t₃). Lock-in l on every new member and on unit–unit mergers. Search uniform, p uncalibrated (varied). Floor entrant is a recapitalized going concern with its record. Relative returns correlated within a market via G (burstiness measured). Floor calibration to the light-tailed families. Large-entrant rejections explained by inference only. Homogeneous block has lower mean q². Event study secondary, matching-sensitive. l, p, Φ unanchored in the literature (presented as dimensions).

## Changes from v1 (C29 → C38)
Base rule loggain → moment. K at α = 0.3: 4.3/4.0/3.7 → 5.5/5.4/4.8; K/K* 0.36/0.25/0.08 → 0.46/0.34/0.10. Pooled capital at α = 0.3: 0.74–0.79 → 0.90–0.94. Voluntary exits ~1 → 0. Hazard ratio "8–10×" → 7–18× (per-status definition). Ordinary-period share: 41/38/18% → 50/38/22%. Panel gap at α = 0.3: 0.43/0.82/1.14 → 0.51/0.85/1.19 pp. Endogenous α 0.69–0.78 → 0.91–0.93, "1.0 in 4–8%" → 12–22%, K 11.2/11.6/8.5 → 16.9/18.0/14.3, ρ(α, volatility) 0.24–0.36 → 0.04–0.10, n 10,691 → 7,630; the α* ≈ K/(K+1) anchor and the birth–death formula K ≈ 1 + j/h are dropped. Equal split K 2.2–2.5 → 2.1–3.2. Hill at α = 0.3: 1.17/1.17/1.10 → 1.21/1.20/1.12. Search K/K* at p = 1: 60/45/14% → 62/50/17%. Cost level: ±0.5 → −0.6 per doubling, ×5–×20 added. Lookback 3.3 → 5.3 (loggain) and 4.8 → 6.6 (moment). Event study: owner path "0 to +0.2" → matching-dependent, post-entry gap +0.01 to +0.26. New: Table 3c decomposition, recruit-size mechanism, approximation/bias quantification, ν block, homogeneous block, fixed α = 0.9 cell. "Attrition, not consent, binds" → "survival first, then what a window can verify; the transfer conflict is immaterial; cost at the margin". Replay rule dropped from the paper.
